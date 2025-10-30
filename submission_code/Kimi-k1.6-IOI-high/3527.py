from bisect import bisect_left, bisect_right
from collections import defaultdict

class DynamicRuns:
    def __init__(self, pairs):
        self.n = len(pairs)
        self.pairs = pairs.copy()
        self.runs = []
        self.freq = defaultdict(int)
        self.false_count = sum(not p for p in pairs)
        self._initialize_runs()
    
    def _initialize_runs(self):
        self.runs.clear()
        self.freq.clear()
        i = 0
        while i < self.n:
            if self.pairs[i]:
                start = i
                while i < self.n and self.pairs[i]:
                    i += 1
                end = i - 1
                self.runs.append((start, end))
                self.freq[end - start + 1] += 1
            else:
                i += 1
        self.runs.sort()
    
    def _find_run_containing(self, idx):
        left = 0
        right = len(self.runs) - 1
        res = -1
        while left <= right:
            mid = (left + right) // 2
            current_start = self.runs[mid][0]
            current_end = self.runs[mid][1]
            if current_start <= idx <= current_end:
                res = mid
                break
            elif idx < current_start:
                right = mid - 1
            else:
                left = mid + 1
        return res
    
    def update_pair(self, idx, new_value):
        old_value = self.pairs[idx]
        if old_value == new_value:
            return
        self.pairs[idx] = new_value
        if new_value:
            self.false_count -= 1
        else:
            self.false_count += 1
        if old_value:
            # Need to split the run if old pair was True
            run_idx = self._find_run_containing(idx)
            if run_idx != -1:
                self._split_run(run_idx, idx)
        else:
            # Merge adjacent runs if new pair is True
            self._merge_pair(idx)
        self.runs.sort()
    
    def _split_run(self, run_idx, split_idx):
        start, end = self.runs[run_idx]
        length = end - start + 1
        self.freq[length] -= 1
        if self.freq[length] == 0:
            del self.freq[length]
        del self.runs[run_idx]
        if start <= split_idx - 1:
            self.freq[split_idx - start] += 1
            self.runs.append((start, split_idx - 1))
        if split_idx + 1 <= end:
            new_length = end - (split_idx + 1) + 1
            self.freq[new_length] += 1
            self.runs.append((split_idx + 1, end))
    
    def _merge_pair(self, idx):
        new_run_start = idx
        new_run_end = idx
        left_idx = self._find_run_containing(idx - 1)
        right_idx = self._find_run_containing(idx + 1)
        if left_idx != -1 and self.pairs[idx - 1]:
            run_start, run_end = self.runs[left_idx]
            self.freq[run_end - run_start + 1] -= 1
            if self.freq[run_end - run_start + 1] == 0:
                del self.freq[run_end - run_start + 1]
            del self.runs[left_idx]
            new_run_start = run_start
        if right_idx != -1 and self.pairs[idx + 1] if idx + 1 < self.n else False:
            new_right_idx = self._find_run_containing(idx + 1)
            if new_right_idx != -1:
                run_start, run_end = self.runs[new_right_idx]
                self.freq[run_end - run_start + 1] -= 1
                if self.freq[run_end - run_start + 1] == 0:
                    del self.freq[run_end - run_start + 1]
                del self.runs[new_right_idx]
                new_run_end = run_end
        new_length = new_run_end - new_run_start + 1
        self.freq[new_length] += 1
        self.runs.append((new_run_start, new_run_end))
    
    def query(self, s):
        if self.false_count == 0:
            return self.n
        required = s - 1
        total = 0
        for length in self.freq:
            if length >= required:
                total += self.freq[length] * (length - required + 1)
        return total

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        pairs = [False] * n
        for i in range(n):
            pairs[i] = colors[i] != colors[(i + 1) % n]
        dr = DynamicRuns(pairs)
        ans = []
        for q in queries:
            if q[0] == 1:
                s = q[1]
                current = dr.query(s)
                ans.append(current)
            else:
                x = q[1]
                new_color = q[2]
                x_prev = (x - 1) % n
                x_curr = x % n
                colors[x] = new_color
                for idx in [x_prev, x_curr]:
                    if 0 <= idx < n:
                        new_pair_val = (colors[idx] != colors[(idx + 1) % n])
                        dr.update_pair(idx, new_pair_val)
        return ans