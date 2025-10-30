from typing import List

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1)

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n < 3:
            return [0] * len([q for q in queries if q[0] == 1])
        
        is_peak = [False] * n
        ft = FenwickTree(n)
        
        # Initialize peaks and Fenwick Tree
        for i in range(1, n - 1):
            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                is_peak[i] = True
                ft.update(i + 1, 1)  # Fenwick is 1-based
        
        answer = []
        
        for q in queries:
            if q[0] == 2:
                idx = q[1]
                val = q[2]
                old_val = nums[idx]
                nums[idx] = val
                
                # Check affected positions: idx-1, idx, idx+1
                affected = [idx - 1, idx, idx + 1]
                for pos in affected:
                    if pos < 1 or pos >= n - 1:
                        continue  # pos must be between 1 and n-2 inclusive
                    
                    prev = nums[pos - 1]
                    curr = nums[pos]
                    next_val = nums[pos + 1]
                    new_is_peak = (curr > prev) and (curr > next_val)
                    old = is_peak[pos]
                    
                    if new_is_peak != old:
                        delta = new_is_peak - old
                        ft.update(pos + 1, delta)
                        is_peak[pos] = new_is_peak
            else:
                l, r = q[1], q[2]
                if l >= r:
                    answer.append(0)
                    continue
                start = l + 1
                end = r - 1
                if start > end:
                    answer.append(0)
                else:
                    # Convert to 1-based indices for Fenwick
                    res = ft.range_query(start + 1, end + 1)
                    answer.append(res)
        
        return answer