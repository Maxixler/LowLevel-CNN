#ifndef MAZE_H
#define MAZE_H

#define SIZE 15
#define ACTIONS 4

// Donanım fonksiyonu prototipi
// grid: Labirent haritası (Girdi)
// q_table: Öğrenilen değerler (Çıktı)
// episodes: Kaç tur eğitim yapılacağı
void q_learning_accel(int grid[SIZE][SIZE], float q_table[SIZE][SIZE][ACTIONS], int episodes);

#endif