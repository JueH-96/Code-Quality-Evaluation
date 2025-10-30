import bisect
from collections import defaultdict

class Solution:
    def numberOfAlternatingGroups(self, colors: list[int], queries: list[list[int]]) -> list[int]:
        n = len(colors)
        runs = []
        if n == 0:
            return []
        
        # Initial runs setup
        current_start = 0
        for i in range(1, n):
            if colors[i] == colors[i-1]:
                runs.append((current_start, i-1))
                current_start = i
        runs.append((current_start, n-1))
        runs.sort()
        start_indices = [run[0] for run in runs]
        
        # Frequency tracking with Fenwick Trees
        class FenwickTree:
            def __init__(self, size):
                self.size = size
                self.tree = [0] * (size + 2)  # 1-based indexing
            
            def update(self, idx, delta):
                idx += 1  # convert to 1-based
                while idx <= self.size + 1:
                    self.tree[idx] += delta
                    idx += idx & -idx
            
            def query(self, idx):
                idx += 1  # convert to 1-based
                res = 0
                while idx > 0:
                    res += self.tree[idx]
                    idx -= idx & -idx
                return res
        
        max_possible = n
        tree1 = FenwickTree(max_possible)  # Tracks sum of (L + 1) * count
        tree2 = FenwickTree(max_possible)  # Tracks sum of count
        
        for run in runs:
            L = run[1] - run[0] + 1
            tree1.update(L, L + 1)
            tree2.update(L, 1)
        
        result = []
        
        for q in queries:
            if q[0] == 1:
                s = q[1]
                if s < 2:
                    result.append(0)
                    continue
                sum_l_plus_1 = tree1.query(max_possible) - tree1.query(s-1)
                sum_freq = tree2.query(max_possible) - tree2.query(s-1)
                result.append(sum_l_plus_1 - s * sum_freq)
            else:
                idx = q[1]
                new_color = q[2]
                if colors[idx] == new_color:
                    continue
                colors[idx] = new_color
                
                # Find affected runs
                affected_indices = {idx-1, idx, idx+1}
                affected = set()
                for pos in affected_indices:
                    pos_mod = pos % n
                    i = bisect.bisect_left(start_indices, pos_mod)
                    if i < len(runs) and runs[i][0] <= pos_mod <= runs[i][1]:
                        affected.add(i)
                    elif i > 0 and runs[i-1][0] <= pos_mod <= runs[i-1][1]:
                        affected.add(i-1)
                
                # Remove affected runs
                removed_runs = []
                for i in sorted(affected, reverse=True):
                    if 0 <= i < len(runs):
                        run = runs[i]
                        L = run[1] - run[0] + 1
                        tree1.update(L, -(L + 1))
                        tree2.update(L, -1)
                        removed_runs.append(run)
                        del runs[i]
                start_indices = [run[0] for run in runs]
                
                # Rebuild runs in the affected area
                if not removed_runs:
                    continue
                min_start = min(run[0] for run in removed_runs)
                max_end = max(run[1] for run in removed_runs)
                # Handle circular wrap-around
                if min_start > max_end:
                    new_runs = []
                    # Process from min_start to n-1
                    current_start = min_start
                    prev_color = colors[min_start]
                    for i in range(min_start, n):
                        if colors[i] == prev_color:
                            if current_start <= i-1:
                                new_runs.append((current_start, i-1))
                            current_start = i
                            prev_color = colors[i]
                        else:
                            prev_color = colors[i]
                    if current_start <= n-1:
                        new_runs.append((current_start, n-1))
                    # Process from 0 to max_end
                    current_start = 0
                    prev_color = colors[0]
                    for i in range(0, max_end + 1):
                        if colors[i] == prev_color:
                            if current_start <= i-1:
                                new_runs.append((current_start, i-1))
                            current_start = i
                            prev_color = colors[i]
                        else:
                            prev_color = colors[i]
                    if current_start <= max_end:
                        new_runs.append((current_start, max_end))
                else:
                    new_runs = []
                    current_start = min_start
                    prev_color = colors[min_start]
                    for i in range(min_start, max_end + 1):
                        if colors[i] == prev_color:
                            if current_start <= i-1:
                                new_runs.append((current_start, i-1))
                            current_start = i
                            prev_color = colors[i]
                        else:
                            prev_color = colors[i]
                    if current_start <= max_end:
                        new_runs.append((current_start, max_end))
                
                # Merge with adjacent runs if possible
                merged_runs = []
                for run in new_runs:
                    if not merged_runs:
                        merged_runs.append(run)
                    else:
                        last_run = merged_runs[-1]
                        if run[0] == last_run[1] + 1 and colors[last_run[1]] != colors[run[0]]:
                            merged_runs[-1] = (last_run[0], run[1])
                        else:
                            merged_runs.append(run)
                
                # Insert new runs
                for run in merged_runs:
                    if run[0] > run[1]:
                        continue
                    L = run[1] - run[0] + 1
                    tree1.update(L, L + 1)
                    tree2.update(L, 1)
                    pos = bisect.bisect_left(start_indices, run[0])
                    runs.insert(pos, run)
                    start_indices.insert(pos, run[0])
        
        return result