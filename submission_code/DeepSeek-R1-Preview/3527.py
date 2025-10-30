import bisect

class FenwickTree:
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_sum = [0] * (max_size + 2)
        self.tree_count = [0] * (max_size + 2)
    
    def update(self, idx, delta_sum, delta_count):
        idx += 1  # To handle 1-based indexing
        while idx <= self.max_size + 1:
            self.tree_sum[idx] += delta_sum
            self.tree_count[idx] += delta_count
            idx += idx & -idx
    
    def query(self, idx):
        idx += 1  # To handle 1-based indexing
        res_sum = 0
        res_count = 0
        while idx > 0:
            res_sum += self.tree_sum[idx]
            res_count += self.tree_count[idx]
            idx -= idx & -idx
        return res_sum, res_count
    
    def range_query(self, a, b):
        sum_a, count_a = self.query(a - 1)
        sum_b, count_b = self.query(b)
        return sum_b - sum_a, count_b - count_a

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        N = len(colors)
        self.N = N
        self.colors = colors.copy()
        self.transitions = []
        for i in range(N):
            next_i = (i + 1) % N
            self.transitions.append(1 if self.colors[i] != self.colors[next_i] else 0)
        
        self.zero_positions = [i for i, t in enumerate(self.transitions) if t == 0]
        self.runs = self.compute_runs()
        self.max_run_length = max(self.runs) if self.runs else 0
        self.ft = FenwickTree(self.max_run_length)
        for run in self.runs:
            if run > 0:
                self.ft.update(run, run, 1)
        
        answer = []
        for q in queries:
            if q[0] == 1:
                K = q[1]
                W = K - 1
                if W < 0:
                    answer.append(0)
                    continue
                if self.zero_positions:
                    sum_L, count = self.ft.range_query(W, self.max_run_length)
                    res = sum_L - (W - 0) * count
                else:
                    if W <= N:
                        res = N - W + 1
                    else:
                        res = 0
                answer.append(res if res > 0 else 0)
            elif q[0] == 2:
                idx = q[1]
                new_color = q[2]
                if self.colors[idx] == new_color:
                    continue
                old_color = self.colors[idx]
                self.colors[idx] = new_color
                prev_idx = (idx - 1) % N
                next_idx = (idx + 1) % N
                affected = [prev_idx, idx]
                for i in affected:
                    old_transition = self.transitions[i]
                    j = (i + 1) % N
                    new_transition = 1 if self.colors[i] != self.colors[j] else 0
                    if new_transition != old_transition:
                        self.transitions[i] = new_transition
                        self.update_runs_and_ft(i, old_transition, new_transition)
        return answer
    
    def compute_runs(self):
        zeros = [i for i, t in enumerate(self.transitions) if t == 0]
        runs = []
        if not zeros:
            if 1 in self.transitions:
                runs.append(self.N)
            return runs
        m = len(zeros)
        for i in range(m):
            current = zeros[i]
            next_ = zeros[(i + 1) % m]
            if next_ > current:
                run_length = next_ - current - 1
            else:
                run_length = (next_ + self.N) - current - 1
            if run_length > 0:
                runs.append(run_length)
        return runs
    
    def update_runs_and_ft(self, i, old_t, new_t):
        pass