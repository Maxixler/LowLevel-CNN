#include <iostream>
#include <vector>
#include <fstream>
#include "maze.h"

// Basit Labirent Oluşturucu (Test için)
void generate_simple_maze(int grid[SIZE][SIZE])
{
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            grid[i][j] = 0; // Hepsi boş

    // Basit engeller ekle
    grid[1][1] = 1;
    grid[2][2] = 1;
    grid[3][3] = 1;
    grid[5][5] = 1;
    grid[5][6] = 1;
    grid[5][7] = 1;
}

int main()
{
    int grid[SIZE][SIZE];
    // Vitis HLS stack boyutu sınırlı olduğu için statik/heap kullanmak iyidir
    static float q_table[SIZE][SIZE][ACTIONS];

    // 1. Veriyi Hazırla
    generate_simple_maze(grid);
    std::cout << "Labirent Hazırlandı. FPGA Fonksiyonu Çağırılıyor..." << std::endl;

    // 2. DONANIM FONKSİYONUNU ÇAĞIR (Simülasyon)
    // 1000 Epizot eğit
    q_learning_accel(grid, q_table, 1000);

    // 3. Sonucu Test Et (Ajanın yolunu bulması)
    std::cout << "Eğitim Tamamlandı. Test Ediliyor..." << std::endl;

    int cx = 0, cy = 0;
    int steps = 0;

    std::cout << "Yol: (0,0)";
    while (steps < 50)
    {
        int best_action = -1;
        float max_q = -1e9;

        // En iyi hareketi Q tablosundan seç
        for (int a = 0; a < 4; a++)
        {
            if (q_table[cy][cx][a] > max_q)
            {
                max_q = q_table[cy][cx][a];
                best_action = a;
            }
        }

        // Koordinatları güncelle
        if (best_action == 0)
            cx++; // Sağ
        else if (best_action == 1)
            cy++; // Aşağı
        else if (best_action == 2)
            cx--; // Sol
        else if (best_action == 3)
            cy--; // Yukarı

        std::cout << " -> (" << cx << "," << cy << ")";

        if (cx == SIZE - 1 && cy == SIZE - 1)
        {
            std::cout << "\n\nBAŞARILI! Hedefe ulaşıldı." << std::endl;
            return 0;
        }
        steps++;
    }

    std::cout << "\n\nBAŞARISIZ! Hedefe ulaşılamadı." << std::endl;
    return 1;
}