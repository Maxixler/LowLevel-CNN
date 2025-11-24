import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque
import random
import time

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

print("🚀 MÜKEMMEL LABİRENT ÇÖZÜCÜ - OPTİMİZE VE DÜZELTİLMİŞ VERSİYON!")

# =============================================================================
# 1. DÜZELTİLMİŞ LABİRENT SİSTEMİ - GARANTİLİ BAĞLANTILI
# =============================================================================

class PerfectMaze:
    def __init__(self, width=15, height=15, obstacle_density=0.1):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        self._generate_perfect_maze(obstacle_density)
        self._precompute_actions()
    
    def _precompute_actions(self):
        """Tüm hücreler için geçerli aksiyonları önceden hesapla"""
        self.valid_actions = np.empty((self.height, self.width), dtype=object)
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:
                    actions = []
                    # Sıralama: sağ(0), aşağı(1), sol(2), yukarı(3)
                    if x + 1 < self.width and self.grid[y, x + 1] == 0:
                        actions.append(0)
                    if y + 1 < self.height and self.grid[y + 1, x] == 0:
                        actions.append(1)
                    if x - 1 >= 0 and self.grid[y, x - 1] == 0:
                        actions.append(2)
                    if y - 1 >= 0 and self.grid[y - 1, x] == 0:
                        actions.append(3)
                    self.valid_actions[y, x] = np.array(actions, dtype=np.int8)
                else:
                    self.valid_actions[y, x] = np.array([], dtype=np.int8)

    def _generate_perfect_maze(self, density):
        """Garantili bağlantılı labirent oluşturma"""
        print(f"🏗️  MÜKEMMEL LABİRENT OLUŞTURULUYOR: {self.width}x{self.height}, Engel: {density:.0%}")
        
        # Önce tamamen boş labirent
        self.grid.fill(0)
        
        # DFS ile labirent oluştur (garantili bağlantı)
        self._dfs_maze_generation()
        
        # Kontrollü engel ekleme (bağlantıyı bozmadan)
        self._add_safe_obstacles(density)
        
        # Son bağlantı kontrolü
        self._ensure_connectivity()
        
        print(f"✅ Mükemmel labirent hazır! Engeller: {np.sum(self.grid)}/{self.width*self.height}")
    
    def _dfs_maze_generation(self):
        """DFS ile labirent oluşturma - garantili bağlantı"""
        visited = np.zeros((self.height, self.width), dtype=bool)
        stack = [(0, 0)]
        visited[0, 0] = True
        
        while stack:
            x, y = stack[-1]
            
            # Komşular (2 adım ara ile)
            neighbors = []
            for dx, dy in [(0,2), (2,0), (0,-2), (-2,0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    not visited[ny, nx]):
                    neighbors.append((nx, ny, dx//2, dy//2))
            
            if neighbors:
                # Rastgele komşu seç
                nx, ny, wx, wy = random.choice(neighbors)
                
                # Aradaki duvarı kaldır
                self.grid[y + wy, x + wx] = 0
                
                # Yeni hücreyi işaretle
                visited[ny, nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()
    
    def _add_safe_obstacles(self, density):
        """Güvenli engel ekleme - bağlantıyı bozmadan"""
        target_obstacles = int(self.width * self.height * density)
        added_obstacles = 0
        max_attempts = target_obstacles * 10
        
        for attempt in range(max_attempts):
            if added_obstacles >= target_obstacles:
                break
                
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            
            # Sadece boş hücrelere ve kenar olmayanlara engel ekle
            if (self.grid[y, x] == 0 and 
                x not in [0, self.width-1] and y not in [0, self.height-1]):
                
                # Geçici olarak engel koy
                self.grid[y, x] = 1
                
                # Bağlantı kontrolü
                if self._is_fully_connected():
                    added_obstacles += 1
                else:
                    # Bağlantıyı bozuyorsa geri al
                    self.grid[y, x] = 0
    
    def _is_fully_connected(self):
        """Labirentin tamamen bağlı olup olmadığını kontrol et"""
        # İlk boş hücreyi bul
        start = None
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:
                    start = (x, y)
                    break
            if start:
                break
        
        if not start:
            return False
            
        # BFS ile bağlantı kontrolü
        visited = np.zeros((self.height, self.width), dtype=bool)
        queue = deque([start])
        visited[start[1], start[0]] = True
        reachable_count = 0
        
        while queue:
            x, y = queue.popleft()
            reachable_count += 1
            
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    not visited[ny, nx] and self.grid[ny, nx] == 0):
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        
        total_empty = np.sum(self.grid == 0)
        return reachable_count == total_empty
    
    def _ensure_connectivity(self):
        """Bağlantıyı garanti et"""
        if not self._is_fully_connected():
            print("⚠️  Labirent bağlantısız, düzeltiliyor...")
            # Basit düzeltme: rastgele engelleri kaldır
            for _ in range(10):
                x, y = random.randint(1, self.width-2), random.randint(1, self.height-2)
                if self.grid[y, x] == 1:
                    self.grid[y, x] = 0
                    if self._is_fully_connected():
                        break
    
    def is_valid_position(self, pos):
        x, y = pos
        return (0 <= x < self.width and 0 <= y < self.height and 
                self.grid[y, x] == 0)
    
    def get_possible_actions(self, pos):
        """Optimize aksiyon getirme"""
        return self.valid_actions[pos[1], pos[0]]
    
    def has_path(self, start, end):
        """İki nokta arasında yol olup olmadığını kontrol et"""
        if not (self.is_valid_position(start) and self.is_valid_position(end)):
            return False
            
        visited = np.zeros((self.height, self.width), dtype=bool)
        queue = deque([start])
        visited[start[1], start[0]] = True
        
        while queue:
            x, y = queue.popleft()
            if (x, y) == end:
                return True
                
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    not visited[ny, nx] and self.grid[ny, nx] == 0):
                    visited[ny, nx] = True
                    queue.append((nx, ny))


# =============================================================================
# 2. MÜKEMMEL HYBRID ÇÖZÜCÜ - OPTİMİZE EDİLMİŞ
# =============================================================================

class PerfectSolver:
    def __init__(self, maze, learning_rate=0.15, discount_factor=0.95, exploration_rate=0.25):
        self.maze = maze
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = 0.08
        self.epsilon_decay = 0.999
        
        self.directions = [(1,0), (0,1), (-1,0), (0,-1)]  # sağ, aşağı, sol, yukarı
        self.visited_positions = set()
        
        # Optimize data structures with NumPy
        # Q-table: (pos_y, pos_x, target_y, target_x, action)
        # Using full coordinates for maximum accuracy (No abstraction)
        self.q_table = np.zeros((self.maze.height, self.maze.width, self.maze.height, self.maze.width, 4), dtype=np.float32)
        
        # Heuristic cache: (pos_y, pos_x, target_y, target_x, action)
        self.heuristic_cache = np.full((self.maze.height, self.maze.width, self.maze.height, self.maze.width, 4), -np.inf, dtype=np.float32)
        
        # İstatistikler
        self.stats = {
            'success': [], 'rewards': [], 'steps': [], 'q_size': [],
            'efficiency': [], 'exploration_rate': [], 'learning_progress': []
        }
        
        print("🤖 MÜKEMMEL ÇÖZÜCÜ AKTİVE! (Full State - Maximum Precision)")
        print(f"   LR: {learning_rate}, γ: {discount_factor}, ε: {exploration_rate}")
    
    def get_heuristic_value(self, position, target, action):
        """Akıllı heuristic - mesafe + keşif + çeşitlilik"""
        py, px = position[1], position[0]
        ty, tx = target[1], target[0]
        
        if self.heuristic_cache[py, px, ty, tx, action] > -1e9:
            return self.heuristic_cache[py, px, ty, tx, action]
        
        dx, dy = self.directions[action]
        new_pos = (px + dx, py + dy)
        
        if not self.maze.is_valid_position(new_pos):
            result = -25.0
        else:
            # Manhattan mesafesi
            current_dist = abs(px - tx) + abs(py - ty)
            new_dist = abs(new_pos[0] - tx) + abs(new_pos[1] - ty)
            
            # Mesafe ödülü
            distance_bonus = (current_dist - new_dist) * 15
            
            # Keşif bonusu
            exploration_bonus = 0 
            
            # Çıkış çeşitliliği bonusu
            possible_actions = len(self.maze.get_possible_actions(new_pos))
            diversity_bonus = min(2, possible_actions * 0.3)
            
            result = distance_bonus + exploration_bonus + diversity_bonus
        
        self.heuristic_cache[py, px, ty, tx, action] = result
        return result
    
    def choose_action(self, position, target, training=True):
        """Akıllı aksiyon seçimi"""
        possible_actions = self.maze.get_possible_actions(position)
        
        if len(possible_actions) == 0:
            return None
        
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if training:
            progress = len(self.stats['success']) / 2000 if self.stats['success'] else 0
            q_weight = min(0.7, 0.3 + progress * 0.4)
            
            if random.random() < (1 - q_weight):
                return self._heuristic_action(position, target, possible_actions)
            else:
                return self._q_learning_action(position, target, possible_actions)
        else:
            return (self._q_learning_action(position, target, possible_actions) 
                    if random.random() < 0.8 else 
                    self._heuristic_action(position, target, possible_actions))
    
    def _heuristic_action(self, position, target, possible_actions):
        """Heuristic aksiyon"""
        # Vectorized heuristic lookup
        action_scores = np.array([self.get_heuristic_value(position, target, a) for a in possible_actions])
        
        # Softmax
        scores = action_scores - np.max(action_scores)
        temperature = max(0.5, 1.0 - self.epsilon)
        exp_scores = np.exp(scores / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return np.random.choice(possible_actions, p=probabilities)
    
    def _q_learning_action(self, position, target, possible_actions):
        """Q-learning aksiyonu - Direct Access Optimized"""
        py, px = position[1], position[0]
        ty, tx = target[1], target[0]
        
        # Get Q-values for all possible actions directly
        q_values = self.q_table[py, px, ty, tx, possible_actions]
        
        # Argmax with random tie-breaking (Optimized)
        max_q = np.max(q_values)
        # np.where is faster than list comprehension for numpy arrays
        indices = np.where(q_values == max_q)[0]
        best_idx = np.random.choice(indices)
        return possible_actions[best_idx]
    
    def update_q_value(self, position, target, action, reward, next_position):
        """Gelişmiş Q-learning update - Direct Access Optimized"""
        py, px = position[1], position[0]
        ty, tx = target[1], target[0]
        
        current_q = self.q_table[py, px, ty, tx, action]
        
        # Next state Q değeri
        next_max_q = 0.0
        next_actions = self.maze.get_possible_actions(next_position)
        if len(next_actions) > 0:
            npy, npx = next_position[1], next_position[0]
            # Direct slice access is much faster
            next_max_q = np.max(self.q_table[npy, npx, ty, tx, next_actions])
        
        # Q-learning update
        td_target = reward + self.gamma * next_max_q
        new_q = current_q + self.lr * (td_target - current_q)
        
        self.q_table[py, px, ty, tx, action] = new_q
    
    def calculate_reward(self, old_pos, new_pos, target, is_done, step, max_steps=80):
        """Optimize reward sistemi"""
        if is_done:
            return 200 + (max_steps - step) * 5
        
        if not self.maze.is_valid_position(new_pos):
            return -20
        
        old_dist = abs(old_pos[0]-target[0]) + abs(old_pos[1]-target[1])
        new_dist = abs(new_pos[0]-target[0]) + abs(new_pos[1]-target[1])
        
        if new_dist < old_dist:
            progress = (old_dist - new_dist) / old_dist if old_dist > 0 else 0
            return 5 + progress * 10
        elif new_dist > old_dist:
            return -3
        else:
            return -1

# =============================================================================
# 3. MÜKEMMEL EĞİTİM SİSTEMİ - CURRICULUM LEARNING
# =============================================================================

def perfect_training(maze_size=15, obstacle_density=0.1, episodes=2000):
    print("🔥 MÜKEMMEL EĞİTİM BAŞLIYOR!")
    print(f"   Labirent: {maze_size}x{maze_size}, Engel: {obstacle_density:.0%}, Episode: {episodes}")
    
    start_time = time.time()
    maze = PerfectMaze(maze_size, maze_size, obstacle_density)
    agent = PerfectSolver(maze)
    
    success_history = deque(maxlen=100)
    
    # Cache methods for performance
    choose_action = agent.choose_action
    update_q_value = agent.update_q_value
    calculate_reward = agent.calculate_reward
    is_valid_position = maze.is_valid_position
    directions = agent.directions
    
    for episode in range(episodes):
        start, target = get_curriculum_points(maze, episode, episodes)
        
        current_pos = start
        total_reward = 0
        steps = 0
        max_steps = maze_size * 4
        success = False
        
        # Visited set reset
        agent.visited_positions.clear()
        agent.visited_positions.add(start)
        
        while steps < max_steps:
            action = choose_action(current_pos, target, training=True)
            if action is None:
                break
            
            dx, dy = directions[action]
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            is_done = (new_pos == target)
            reward = calculate_reward(current_pos, new_pos, target, is_done, steps, max_steps)
            
            if is_valid_position(new_pos):
                update_q_value(current_pos, target, action, reward, new_pos)
                current_pos = new_pos
                agent.visited_positions.add(new_pos)
            
            total_reward += reward
            steps += 1
            
            if is_done:
                success = True
                break
        
        # İstatistikleri güncelle
        optimal_steps = manhattan_distance(start, target)
        efficiency = optimal_steps / steps if success and steps > 0 else 0
        
        success_history.append(success)
        
        agent.stats['success'].append(success)
        agent.stats['rewards'].append(total_reward)
        agent.stats['steps'].append(steps)
        agent.stats['q_size'].append(agent.q_table.size) # Size in elements
        agent.stats['efficiency'].append(efficiency)
        agent.stats['exploration_rate'].append(agent.epsilon)
        agent.stats['learning_progress'].append(np.mean(success_history) if success_history else 0)
        
        if (episode + 1) % 200 == 0:
            recent_success = np.mean(success_history)
            recent_steps = np.mean(agent.stats['steps'][-100:])
            recent_efficiency = np.mean([e for e in agent.stats['efficiency'][-100:] if e > 0])
            
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            
            print(f"📍 {episode+1:4d} | "
                  f"Başarı: {recent_success:.3f} | "
                  f"Verim: {recent_efficiency:.1%} | "
                  f"Adım: {recent_steps:5.1f} | "
                  f"ϵ: {agent.epsilon:.3f} | "
                  f"Hız: {eps_per_sec:.1f} eps/s")
            
            if (recent_success > 0.92 and episode > 1000 and 
                agent.epsilon <= agent.epsilon_min * 1.2):
                print(f"🎯 OPTİMUM BAŞARI! {episode+1}. episode'da durduruldu")
                break
    
    total_time = time.time() - start_time
    print(f"✅ MÜKEMMEL EĞİTİM TAMAMLANDI! Süre: {total_time:.1f}s")
    
    plot_perfect_stats(agent.stats, maze_size)
    return agent, maze

def get_curriculum_points(maze, episode, total_episodes):
    """Curriculum learning ile nokta seçimi"""
    progress = episode / total_episodes
    
    if progress < 0.4:  # İlk %40: Yakın mesafe
        max_dist = maze.width // 3
    elif progress < 0.7:  # Sonraki %30: Orta mesafe
        max_dist = maze.width // 2
    else:  # Son %30: Uzak mesafe
        max_dist = maze.width * 2 // 3
    
    return get_guaranteed_points(maze, max_dist)

def get_guaranteed_points(maze, max_distance=5, min_distance=2):
    """Daha akıllı nokta seçimi - garantili ulaşılabilir noktalar"""
    # Labirentteki tüm boş hücreleri al
    empty_cells = []
    for y in range(maze.height):
        for x in range(maze.width):
            if maze.grid[y, x] == 0:
                empty_cells.append((x, y))
    
    if len(empty_cells) < 2:
        return (0,0), (maze.width-1, maze.height-1)
    
    # Öncelikle yakın mesafeli ulaşılabilir noktalar
    for _ in range(100):
        start, target = random.sample(empty_cells, 2)
        distance = manhattan_distance(start, target)
        
        if (min_distance <= distance <= max_distance and 
            maze.has_path(start, target) and 
            start != target):
            return start, target
    
    # Uzun mesafe deneme
    for _ in range(50):
        start, target = random.sample(empty_cells, 2)
        if maze.has_path(start, target) and start != target:
            return start, target
    
    return empty_cells[0], empty_cells[-1]
def perfect_test(agent, maze, num_tests=8):
    print("\n" + "="*60)
    print("🎯 MÜKEMMEL TEST SÜRECİ")
    print("="*60)
    
    test_cases = []
    
    # 1. Önce kesin ulaşılabilir yakın noktalar
    for i in range(3):
        start, target = get_guaranteed_points(maze, max_distance=8, min_distance=4)
        test_cases.append((start, target, f"YAKIN TEST {i+1}"))
    
    # 2. Orta mesafeli testler
    for i in range(3):
        start, target = get_guaranteed_points(maze, max_distance=15, min_distance=8)
        test_cases.append((start, target, f"ORTA TEST {i+1}"))
    
    # 3. Köşe testlerini sona bırak
    corner_tests = [
        ((0, 0), (maze.width-1, maze.height-1), "SOL-ÜST → SAĞ-ALT"),
        ((0, maze.height-1), (maze.width-1, 0), "SOL-ALT → SAĞ-ÜST")
    ]
    
    for start, target, desc in corner_tests:
        if maze.has_path(start, target):
            test_cases.append((start, target, desc))
        else:
            print(f"⚠️  {desc}: Ulaşılamaz, atlanıyor")
    
    results = []
    successful_tests = 0
    
    for start, target, desc in test_cases:
        print(f"\n🔍 {desc}: {start} → {target}")
        
        path = find_improved_path(agent, maze, start, target, max_steps=maze.width * 5)
        
        if path and path[-1] == target:
            steps = len(path) - 1
            optimal = find_optimal_path(maze, start, target)
            efficiency = optimal / steps if steps > 0 else 0
            
            print(f"   ✅ BAŞARILI | {steps} adım | Optimal: {optimal} | Verim: {efficiency:.1%}")
            results.append((desc, True, steps, optimal, efficiency))
            successful_tests += 1
            
            if successful_tests <= 2:
                visualize_solution(maze, path, start, target, f"{desc}\n{steps} adım")
        else:
            print(f"   ❌ BAŞARISIZ")
            results.append((desc, False, 0, 0, 0))
    
    return results

def find_improved_path(agent, maze, start, target, max_steps=None):
    """Gelişmiş yol bulma - daha akıllı backtracking"""
    if max_steps is None:
        max_steps = maze.width * 4
    
    current_pos = start
    path = [start]
    visited = set([start])
    backtrack_count = 0
    max_backtracks = 5
    
    for step in range(max_steps):
        if current_pos == target:
            return path
        
        action = agent.choose_action(current_pos, target, training=False)
        if action is None:
            break
        
        dx, dy = agent.directions[action]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        if maze.is_valid_position(new_pos):
            if new_pos in visited:
                backtrack_count += 1
                if backtrack_count > max_backtracks and len(path) > 3:
                    backtrack_steps = random.randint(2, min(4, len(path) - 1))
                    path = path[:-backtrack_steps]
                    current_pos = path[-1] if path else start
                    continue
                else:
                    continue
            
            backtrack_count = 0
            current_pos = new_pos
            path.append(current_pos)
            visited.add(current_pos)
        else:
            break
    
    return path

def find_optimal_path(maze, start, end):
    """A* ile optimal yol uzunluğu"""
    open_set = []
    heapq.heappush(open_set, (0, start))
    g_score = {start: 0}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == end:
            return g_score[current]
        
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if (0 <= neighbor[0] < maze.width and 0 <= neighbor[1] < maze.height and 
                maze.grid[neighbor[1], neighbor[0]] == 0):
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + manhattan_distance(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
    
    return float('inf')

def visualize_solution(maze, path, start, target, title):
    """Görselleştirme"""
    plt.figure(figsize=(10, 10))
    
    plt.imshow(maze.grid, cmap='binary', origin='upper')
    
    if len(path) > 1:
        path_array = np.array(path)
        plt.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=3, alpha=0.7)
        plt.plot(path_array[:, 0], path_array[:, 1], 'go', markersize=4, alpha=0.5)
    
    plt.plot(start[0], start[1], 'bs', markersize=15, label='Başlangıç')
    plt.plot(target[0], target[1], 'r*', markersize=20, label='Hedef')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(title, fontsize=14, fontweight='bold')
    plt.show()

def plot_perfect_stats(stats, maze_size):
    """İstatistik grafikleri"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    episodes = len(stats['success'])
    
    # Başarı oranı
    window = min(100, episodes // 10)
    if episodes > window:
        success_ma = np.convolve(stats['success'], np.ones(window)/window, mode='valid')
        ax1.plot(success_ma, 'b-', linewidth=2)
    ax1.set_title('Başarı Oranı', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Başarı Oranı')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Verimlilik
    if episodes > window:
        efficiency_clean = [e for e in stats['efficiency'] if e > 0]
        if efficiency_clean:
            eff_ma = np.convolve(efficiency_clean, np.ones(window)/window, mode='valid')
            ax2.plot(eff_ma, 'g-', linewidth=2)
    ax2.set_title('Verimlilik', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Verimlilik')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Adım sayısı
    ax3.plot(stats['steps'], 'r-', alpha=0.6, linewidth=1)
    ax3.set_title('Adım Sayısı', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Adım')
    ax3.grid(True, alpha=0.3)
    
    # Q-table boyutu
    ax4.plot(stats['q_size'], 'purple', alpha=0.8)
    ax4.set_title('Q-table Boyutu (Eleman)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('State Sayısı')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Mükemmel Labirent Çözücü Performansı ({maze_size}x{maze_size})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 5. ÇOKLU DENEY SİSTEMİ
# =============================================================================

def run_perfect_experiments():
    """Mükemmel deneyler çalıştır"""
    print("🧪 MÜKEMMEL DENEYLER BAŞLIYOR!")
    
    experiments = [
        {"size": 12, "density": 0.08, "episodes": 1500, "name": "KOLAY"},
        {"size": 15, "density": 0.1, "episodes": 2000, "name": "ORTA"},
        {"size": 18, "density": 0.12, "episodes": 2500, "name": "ZOR"}
    ]
    
    all_results = []
    
    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"🧪 DENEY: {exp['name']}")
        print(f"   Boyut: {exp['size']}x{exp['size']}, Engel: {exp['density']:.0%}")
        print(f"   Episode: {exp['episodes']}")
        print(f"{'='*50}")
        
        try:
            agent, maze = perfect_training(
                maze_size=exp['size'],
                obstacle_density=exp['density'],
                episodes=exp['episodes']
            )
            
            test_results = perfect_test(agent, maze, num_tests=6)
            
            # Sonuç analizi
            successful = [r for r in test_results if r[1]]
            success_rate = len(successful) / len(test_results) if test_results else 0
            
            if successful:
                avg_efficiency = np.mean([r[4] for r in successful])
                avg_steps = np.mean([r[2] for r in successful])
            else:
                avg_efficiency = 0
                avg_steps = 0
            
            exp_result = {
                'name': exp['name'],
                'size': exp['size'],
                'density': exp['density'],
                'success_rate': success_rate,
                'avg_efficiency': avg_efficiency,
                'avg_steps': avg_steps,
                'q_size': agent.q_table.size
            }
            
            all_results.append(exp_result)
            
            print(f"\n📊 {exp['name']} SONUÇ:")
            print(f"   ✅ Başarı Oranı: {success_rate:.1%}")
            print(f"   📈 Verimlilik: {avg_efficiency:.1%}")
            print(f"   👣 Ortalama Adım: {avg_steps:.1f}")
            
        except Exception as e:
            print(f"❌ Deney hatası: {e}")
            continue
    
    # Özet
    print(f"\n{'='*60}")
    print("🎯 TÜM DENEY SONUÇLARI")
    print(f"{'='*60}")
    
    for result in all_results:
        status = "🎉 MÜKEMMEL" if result['success_rate'] >= 0.9 else "✅ İYİ" if result['success_rate'] >= 0.7 else "⚠️  ORTA"
        print(f"🔬 {result['name']:6} | Boyut: {result['size']}x{result['size']} | "
              f"Başarı: {result['success_rate']:6.1%} | Verim: {result['avg_efficiency']:6.1%} | "
              f"Adım: {result['avg_steps']:5.1f} | {status}")
    
    return all_results

# =============================================================================
# 6. ANA PROGRAM - MÜKEMMEL VERSİYON
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🚀 MÜKEMMEL LABİRENT ÇÖZÜCÜ - OPTİMİZE VE DÜZELTİLMİŞ")
    print("="*70)
    
    try:
        # Otomatik başlatma - 15x15 Labirent
        print("⚡ OTOMATİK BAŞLATILIYOR: 15x15 Labirent")
        agent, maze = perfect_training(maze_size=15, obstacle_density=0.1, episodes=2000)
        
        # Testleri çalıştır
        test_results = perfect_test(agent, maze)
        
        # Sonuçları göster
        successful = [r for r in test_results if r[1]]
        if successful:
            success_rate = len(successful) / len(test_results)
            print(f"\n🎯 TEST SONUCU: {success_rate:.1%} başarı")
            
    except KeyboardInterrupt:
        print("\n⏹️  Program kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
    
    print("\n🎉 PROGRAM TAMAMLANDI!")