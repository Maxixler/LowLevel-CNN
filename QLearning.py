import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque
import random

print("🚀 RADİKAL ÇÖZÜM - HYBRID APPROACH!")

# =============================================================================
# 1. SÜPER LABİRENT - NEREDEYSE ENGELSİZ
# =============================================================================

class SuperMaze:
    def __init__(self, width=10, height=10, obstacle_density=0.05):  # %5 engel
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self._generate_super_maze(obstacle_density)
    
    def _generate_super_maze(self, density):
        """Süper basit labirent"""
        print("🏗️  SÜPER LABİRENT OLUŞTURULUYOR...")
        
        # Minimum engel
        for i in range(self.height):
            for j in range(self.width):
                if random.random() < density:
                    self.grid[i, j] = 1
        
        # Tüm önemli noktaları temizle
        for i in range(self.height):
            for j in range(self.width):
                if (i == 0 or j == 0 or i == self.height-1 or j == self.width-1 or 
                    i == j or i + j == self.height-1):
                    self.grid[i, j] = 0
        
        print(f"✅ Süper labirent hazır! Engeller: {np.sum(self.grid)}")
        self.visualize_maze()
    
    def visualize_maze(self):
        """Labirenti göster"""
        plt.figure(figsize=(6, 6))
        plt.imshow(self.grid, cmap='binary', origin='upper')
        plt.title("Labirent - Beyaz: Boş, Siyah: Engel")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < self.height and 0 <= y < self.width and self.grid[x, y] == 0
    
    def get_possible_actions(self, pos):
        """Geçerli aksiyonları getir - hedefe yönelik"""
        actions = []
        directions = [(0,1), (1,0), (0,-1), (-1,0)]  # sağ, aşağı, sol, yukarı
        
        for action, (dx, dy) in enumerate(directions):
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self.is_valid_position(new_pos):
                actions.append(action)
        
        return actions

# =============================================================================
# 2. HYBRID ÇÖZÜCÜ - Q-learning + Heuristic
# =============================================================================

class HybridSolver:
    def __init__(self, maze, learning_rate=0.2, discount_factor=0.9, exploration_rate=0.3):
        self.maze = maze
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = 0.1  # Daha yüksek min exploration
        self.epsilon_decay = 0.998
        
        # Q-tablosu
        self.q_table = {}
        
        # İstatistikler
        self.stats = {
            'success': [],
            'rewards': [],
            'steps': [],
            'q_size': []
        }
        
        print("🤖 HYBRID ÇÖZÜCÜ OLUŞTURULDU!")
        print(f"   Learning Rate: {learning_rate}, Gamma: {discount_factor}")
        print(f"   Exploration: {exploration_rate} -> {self.epsilon_min}")
    
    def get_state_key(self, position, target):
        """Basit state key"""
        return f"{position[0]},{position[1]}|{target[0]},{target[1]}"
    
    def get_heuristic_value(self, position, target, action):
        """Heuristic değer - hedefe yaklaşma puanı"""
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        dx, dy = directions[action]
        new_pos = (position[0] + dx, position[1] + dy)
        
        if not self.maze.is_valid_position(new_pos):
            return -10  # Duvara çarpma
        
        # Manhattan mesafesi iyileşmesi
        current_dist = abs(position[0]-target[0]) + abs(position[1]-target[1])
        new_dist = abs(new_pos[0]-target[0]) + abs(new_pos[1]-target[1])
        
        return (current_dist - new_dist) * 5  # Büyük ödül
    
    def get_q_value(self, state, action, position, target):
        """Q değeri + heuristic"""
        if state not in self.q_table:
            # Heuristic ile başlangıç değeri
            heuristic_val = self.get_heuristic_value(position, target, action)
            self.q_table[state] = [heuristic_val * 0.1] * 4  # Heuristic tabanlı başlangıç
        
        return self.q_table[state][action]
    
    def choose_action(self, position, target):
        """Intelligent action selection"""
        state = self.get_state_key(position, target)
        possible_actions = self.maze.get_possible_actions(position)
        
        if not possible_actions:
            return None
        
        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # %30 ihtimalle heuristic, %70 ihtimalle Q-learning
        if random.random() < 0.3:
            # Heuristic-based action
            action_scores = []
            for action in possible_actions:
                score = self.get_heuristic_value(position, target, action)
                action_scores.append(score)
            
            # Softmax selection
            scores = np.array(action_scores)
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / np.sum(exp_scores)
            return np.random.choice(possible_actions, p=probabilities)
        
        # Q-learning with exploration
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        # Greedy Q-action
        q_values = [self.get_q_value(state, action, position, target) for action in possible_actions]
        return possible_actions[np.argmax(q_values)]
    
    def update_q_value(self, position, target, action, reward, next_position):
        """Aggressive Q-learning update"""
        state = self.get_state_key(position, target)
        next_state = self.get_state_key(next_position, target)
        
        current_q = self.get_q_value(state, action, position, target)
        
        # Next state Q value
        next_max_q = 0
        next_actions = self.maze.get_possible_actions(next_position)
        if next_actions and next_state in self.q_table:
            next_max_q = max([self.q_table[next_state][a] for a in next_actions])
        
        # Q-learning with optimistic initialization
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        
        # Update with momentum
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 4
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, old_pos, new_pos, target, is_done, step):
        """Very aggressive reward system"""
        if is_done:
            return 1000.0  # Çok büyük ödül
        
        if not self.maze.is_valid_position(new_pos):
            return -20.0  # Orta ceza
        
        # Mesafe bazlı ödül (çok agresif)
        old_dist = abs(old_pos[0]-target[0]) + abs(old_pos[1]-target[1])
        new_dist = abs(new_pos[0]-target[0]) + abs(new_pos[1]-target[1])
        
        if new_dist < old_dist:
            return 10.0  # Büyük yaklaşma ödülü
        elif new_dist > old_dist:
            return -5.0  # Uzaklaşma cezası
        else:
            return -1.0  # Küçük ceza

# =============================================================================
# 3. AGGRESSIVE EĞİTİM
# =============================================================================

def aggressive_training(episodes=3000):
    print("🔥 AGGRESSIVE EĞİTİM BAŞLIYOR!")
    
    maze = SuperMaze(10, 10, 0.03)  # Sadece %3 engel
    agent = HybridSolver(maze, learning_rate=0.2, discount_factor=0.9, exploration_rate=0.3)
    
    for episode in range(episodes):
        # FOCUSED TRAINING: Çoğunlukla köşeler
        if random.random() < 0.7:  # %70 ihtimalle köşeler
            points = [(0,0), (9,9), (0,9), (9,0)]
            start, target = random.sample(points, 2)
        else:  # %30 ihtimalle rastgele
            start, target = get_valid_points(maze)
        
        current_pos = start
        total_reward = 0
        steps = 0
        max_steps = 50  # Daha kısa süre
        success = False
        
        while steps < max_steps and not success:
            action = agent.choose_action(current_pos, target)
            if action is None:
                break
            
            directions = [(0,1), (1,0), (0,-1), (-1,0)]
            dx, dy = directions[action]
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            is_done = (new_pos == target)
            reward = agent.calculate_reward(current_pos, new_pos, target, is_done, steps)
            
            if maze.is_valid_position(new_pos):  # DÜZELTME: self yerine maze
                agent.update_q_value(current_pos, target, action, reward, new_pos)
                current_pos = new_pos
            
            total_reward += reward
            steps += 1
            
            if is_done:
                success = True
                break
        
        # İstatistik
        agent.stats['success'].append(1 if success else 0)
        agent.stats['rewards'].append(total_reward)
        agent.stats['steps'].append(steps)
        agent.stats['q_size'].append(len(agent.q_table))
        
        # SIKI TAKİP
        if (episode + 1) % 100 == 0:
            recent_success = np.mean(agent.stats['success'][-100:])
            recent_reward = np.mean(agent.stats['rewards'][-100:])
            recent_steps = np.mean(agent.stats['steps'][-100:])
            
            print(f"📍 {episode+1:4d} | "
                  f"Başarı: {recent_success:.3f} | "
                  f"Ödül: {recent_reward:6.1f} | "
                  f"Adım: {recent_steps:4.1f} | "
                  f"ϵ: {agent.epsilon:.3f} | "
                  f"Q: {len(agent.q_table)}")
            
            # Early success detection
            if recent_success > 0.95 and episode > 500:
                print(f"🎯 EARLY SUCCESS! {episode+1}. episode'da durduruldu")
                break
    
    print("✅ AGGRESSIVE EĞİTİM TAMAMLANDI!")
    plot_aggressive_stats(agent.stats)
    return agent, maze

def get_valid_points(maze):
    """Geçerli noktalar bul"""
    attempts = 0
    while attempts < 50:
        start = (random.randint(0,9), random.randint(0,9))
        target = (random.randint(0,9), random.randint(0,9))
        if (maze.is_valid_position(start) and maze.is_valid_position(target) and 
            start != target):
            return start, target
        attempts += 1
    return (0,0), (9,9)

def plot_aggressive_stats(stats):
    """İstatistikleri göster"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    window = 100
    if len(stats['success']) > window:
        success_ma = [np.mean(stats['success'][i:i+window]) for i in range(len(stats['success'])-window)]
        ax1.plot(success_ma)
    ax1.set_title('Başarı Oranı')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    
    if len(stats['rewards']) > window:
        reward_ma = [np.mean(stats['rewards'][i:i+window]) for i in range(len(stats['rewards'])-window)]
        ax2.plot(reward_ma)
    ax2.set_title('Ortalama Ödül')
    ax2.grid(True)
    
    if len(stats['steps']) > window:
        steps_ma = [np.mean(stats['steps'][i:i+window]) for i in range(len(stats['steps'])-window)]
        ax3.plot(steps_ma)
    ax3.set_title('Ortalama Adım')
    ax3.grid(True)
    
    ax4.plot(stats['q_size'])
    ax4.set_title('Q-table Boyutu')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. SUPER TEST
# =============================================================================

def super_test(agent, maze):
    print("\n" + "="*50)
    print("🎯 SUPER TEST SÜRECİ")
    print("="*50)
    
    test_cases = [
        ((0, 0), (9, 9), "Köşeden Köşeye"),
        ((2, 2), (7, 7), "Çapraz Yol"),
        ((0, 5), (9, 5), "Dikey Geçiş"),
        ((5, 0), (5, 9), "Yatay Geçiş"),
        ((1, 1), (8, 8), "Uzun Çapraz"),
        ((3, 6), (6, 3), "Ters Çapraz"),
        ((2, 3), (7, 6), "Rastgele 1"),
        ((4, 1), (8, 7), "Rastgele 2")
    ]
    
    results = []
    
    for start, target, desc in test_cases:
        if not (maze.is_valid_position(start) and maze.is_valid_position(target)):
            print(f"❌ {desc}: Geçersiz pozisyon")
            continue
            
        print(f"\n🔍 {desc}: {start} → {target}")
        
        path = find_smart_path(agent, maze, start, target)
        
        if path and path[-1] == target:
            steps = len(path) - 1
            optimal = abs(start[0]-target[0]) + abs(start[1]-target[1])
            efficiency = optimal / steps if steps > 0 else 0
            
            print(f"   ✅ BAŞARILI | {steps} adım | Optimal: {optimal} | Verim: {efficiency:.1%}")
            results.append((desc, True, steps, optimal, efficiency))
        else:
            print(f"   ❌ BAŞARISIZ")
            results.append((desc, False, 0, 0, 0))
    
    return results

def find_smart_path(agent, maze, start, target):
    """Akıllı yol bulma"""
    current_pos = start
    path = [start]
    visited = set([start])
    max_steps = 100
    
    for step in range(max_steps):
        if current_pos == target:
            return path
        
        action = agent.choose_action(current_pos, target)
        if action is None:
            break
        
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        dx, dy = directions[action]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        if maze.is_valid_position(new_pos):
            # Döngü önleme
            if new_pos in visited and len(path) > 5:
                # Backtrack
                if len(path) > 2:
                    path.pop()
                    current_pos = path[-1]
                continue
                
            current_pos = new_pos
            path.append(current_pos)
            visited.add(current_pos)
        else:
            break
    
    return path

# =============================================================================
# 5. ANA PROGRAM
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 RADİKAL ÇÖZÜM - HYBRID APPROACH")
    print("="*60)
    
    agent, maze = aggressive_training(episodes=3000)
    test_results = super_test(agent, maze)
    
    print("\n" + "="*60)
    print("📊 RADİKAL SONUÇLAR")
    print("="*60)
    
    successful = [r for r in test_results if r[1]]
    
    if successful:
        success_rate = len(successful) / len(test_results)
        avg_eff = np.mean([r[4] for r in successful])
        
        print(f"🎯 BAŞARI: {success_rate:.1%} ({len(successful)}/{len(test_results)})")
        print(f"📈 VERİMLİLİK: {avg_eff:.1%}")
        
        if success_rate >= 0.9:
            print("🎉 SÜPER! Mükemmel sonuç!")
        elif success_rate >= 0.7:
            print("👍 ÇOK İYİ! Büyük gelişme!")
        elif success_rate >= 0.5:
            print("💪 İYİ! Kabul edilebilir.")
        else:
            print("⚠️  ORTA! Daha iyi olabilir.")
    else:
        print("❌ TESTLERDE BAŞARILI OLUNAMADI!")
    
    print(f"🤖 Q-table states: {len(agent.q_table)}")
    print(f"🎯 Final epsilon: {agent.epsilon:.3f}")