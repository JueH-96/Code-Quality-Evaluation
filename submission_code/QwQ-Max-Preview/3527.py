from typing import List
from collections import defaultdict

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        valid_pairs = [0] * n
        for i in range(n):
            valid_pairs[i] = 0 if colors[i] == colors[(i + 1) % n] else 1
        
        zero_positions = []
        for i in range(n):
            if valid_pairs[i] == 0:
                zero_positions.append(i)
        zero_positions.sort()
        
        next_zero = [0] * n
        if not zero_positions:
            next_zero = [n] * n
        else:
            for i in range(n):
                found = False
                for j in zero_positions:
                    if j >= i:
                        next_zero[i] = j
                        found = True
                        break
                if not found:
                    next_zero[i] = zero_positions[0]
        
        window_size = [0] * n
        if not zero_positions:
            for i in range(n):
                window_size[i] = n
        else:
            for i in range(n):
                j = next_zero[i]
                if j >= i:
                    window_size[i] = j - i
                else:
                    window_size[i] = (j + n - i)
        
        freq = defaultdict(int)
        for size in window_size:
            freq[size] += 1
        
        max_size = max(freq.keys()) if freq else 0
        fenwick_size = max_size + 2
        fenwick = [0] * fenwick_size
        
        def update_fenwick(idx, delta):
            while idx <= max_size:
                fenwick[idx] += delta
                idx += idx & -idx
        
        def query_fenwick(idx):
            res = 0
            while idx > 0:
                res += fenwick[idx]
                idx -= idx & -idx
            return res
        
        for size in freq:
            update_fenwick(size, freq[size])
        
        answer = []
        for query in queries:
            if query[0] == 1:
                s = query[1]
                w = s - 1
                if w < 0 or w > n:
                    answer.append(0)
                    continue
                if not zero_positions:
                    if w <= n:
                        answer.append(n)
                    else:
                        answer.append(0)
                else:
                    total = query_fenwick(max_size) - query_fenwick(w - 1)
                    answer.append(total)
            else:
                idx = query[1]
                new_color = query[2]
                if colors[idx] == new_color:
                    answer.append(-1)
                    continue
                
                prev_color = colors[idx]
                colors[idx] = new_color
                prev_idx = (idx - 1) % n
                next_idx = (idx + 1) % n
                
                for i in [prev_idx, idx]:
                    old_valid = valid_pairs[i]
                    current_color = colors[i]
                    next_color = colors[(i + 1) % n]
                    valid_pairs[i] = 0 if current_color == next_color else 1
                    new_valid = valid_pairs[i]
                    if old_valid == new_valid:
                        continue
                    
                    if old_valid == 0 and new_valid == 1:
                        if i in zero_positions:
                            zero_positions.remove(i)
                    elif old_valid == 1 and new_valid == 0:
                        bisect.insort(zero_positions, i)
                
                zero_positions.sort()
                new_zero_positions = []
                for i in range(n):
                    if valid_pairs[i] == 0:
                        new_zero_positions.append(i)
                new_zero_positions.sort()
                zero_positions = new_zero_positions
                
                new_next_zero = [0] * n
                if not zero_positions:
                    new_next_zero = [n] * n
                else:
                    for i in range(n):
                        found = False
                        for j in zero_positions:
                            if j >= i:
                                new_next_zero[i] = j
                                found = True
                                break
                        if not found:
                            new_next_zero[i] = zero_positions[0]
                
                new_window_size = [0] * n
                if not zero_positions:
                    new_window_size = [n] * n
                else:
                    for i in range(n):
                        j = new_next_zero[i]
                        if j >= i:
                            new_window_size[i] = j - i
                        else:
                            new_window_size[i] = (j + n - i)
                
                for i in range(n):
                    old_size = window_size[i]
                    new_size = new_window_size[i]
                    if old_size != new_size:
                        if freq[old_size] > 0:
                            freq[old_size] -= 1
                            update_fenwick(old_size, -1)
                            if freq[old_size] == 0:
                                del freq[old_size]
                        freq[new_size] += 1
                        update_fenwick(new_size, 1)
                window_size = new_window_size.copy()
                answer.append(-1)
        
        return [ans for ans in answer if ans != -1]