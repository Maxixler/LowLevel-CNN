#include "maze.h"

// Basit bir Pseudo-Random Sayı Üreteci (Xorshift)
// FPGA'de std::random yoktur, bit kaydırma ile rastgelelik üretiriz.
static unsigned int rng_state = 123456789;
unsigned int get_random()
{
#pragma HLS INLINE
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// Q-Learning Çekirdeği
void q_learning_accel(int grid[SIZE][SIZE], float q_table[SIZE][SIZE][ACTIONS], int episodes)
{
// Arayüz Tanımları: DDR hafızasına erişim için AXI Master
#pragma HLS INTERFACE m_axi port = grid bundle = gmem0
#pragma HLS INTERFACE m_axi port = q_table bundle = gmem1
// Kontrol sinyalleri (Start/Stop)
#pragma HLS INTERFACE s_axilite port = episodes
#pragma HLS INTERFACE s_axilite port = return

    // Yerel (On-Chip) Hafıza - Hız için BRAM kullanımı
    float local_q[SIZE][SIZE][ACTIONS];
    int local_grid[SIZE][SIZE];

// Performans için Q tablosunu parçalara böl (Aynı anda okuma/yazma için)
#pragma HLS ARRAY_PARTITION variable = local_q dim = 3 complete

// 1. Veriyi DDR'dan FPGA içine çek (Burst Read)
load_grid:
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
#pragma HLS PIPELINE
            local_grid[i][j] = grid[i][j];
            // Q tablosunu sıfırla
            for (int k = 0; k < ACTIONS; k++)
                local_q[i][j][k] = 0.0f;
        }
    }

    // Parametreler
    float lr = 0.1f;
    float gamma = 0.95f;
    float epsilon = 1.0f;
    float decay = 0.995f;

    // Hareket Vektörleri (Sağ, Aşağı, Sol, Yukarı)
    const int dx[4] = {1, 0, -1, 0};
    const int dy[4] = {0, 1, 0, -1};
    int target_x = SIZE - 1;
    int target_y = SIZE - 1;

// EĞİTİM DÖNGÜSÜ
training_loop:
    for (int ep = 0; ep < episodes; ep++)
    {
        // Bu döngünün pipelining yapılmasını istemiyoruz çünkü
        // her adım bir öncekine bağlı.

        int cx = 0; // Başlangıç X
        int cy = 0; // Başlangıç Y
        int steps = 0;

    // Bir epizot (ajan hedefe varana kadar)
    episode_step:
        while (steps < SIZE * SIZE * 2)
        {
#pragma HLS PIPELINE II = 4
            // II=4: Floating point işlemleri biraz zaman alır, her 4 clockta bir adım.

            if (cx == target_x && cy == target_y)
                break;

            // 1. Eylem Seç (Epsilon-Greedy)
            int action = 0;
            unsigned int rnd = get_random();
            float rnd_f = (float)(rnd % 1000) / 1000.0f;

            if (rnd_f < epsilon)
            {
                action = rnd % 4; // Rastgele
            }
            else
            {
                // Max Q değerini bul
                float max_val = -1e9;
                for (int a = 0; a < 4; a++)
                {
                    if (local_q[cy][cx][a] > max_val)
                    {
                        max_val = local_q[cy][cx][a];
                        action = a;
                    }
                }
            }

            // 2. Hareket Et
            int nx = cx + dx[action];
            int ny = cy + dy[action];

            // Duvar kontrolü ve sınır kontrolü
            if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE || local_grid[ny][nx] == 1)
            {
                nx = cx; // Hareket geçersiz, yerinde kal
                ny = cy;
            }

            // 3. Ödül Hesapla
            float reward = -1.0f;
            if (nx == target_x && ny == target_y)
                reward = 100.0f;
            else if (nx == cx && ny == cy)
                reward = -5.0f; // Duvara çarpma cezası

            // 4. Q Güncellemesi (Bellman Denklemi)
            float max_next_q = -1e9;
            for (int a = 0; a < 4; a++)
            {
                if (local_q[ny][nx][a] > max_next_q)
                    max_next_q = local_q[ny][nx][a];
            }
            if (max_next_q == -1e9)
                max_next_q = 0.0f;

            local_q[cy][cx][action] = local_q[cy][cx][action] + lr * (reward + gamma * max_next_q - local_q[cy][cx][action]);

            // Konumu güncelle
            cx = nx;
            cy = ny;
            steps++;
        }

        // Epsilon Decay
        if (epsilon > 0.01f)
            epsilon *= decay;
    }

// 3. Sonuçları DDR'a geri yaz
write_back:
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            for (int k = 0; k < ACTIONS; k++)
            {
#pragma HLS PIPELINE
                q_table[i][j][k] = local_q[i][j][k];
            }
        }
    }
}