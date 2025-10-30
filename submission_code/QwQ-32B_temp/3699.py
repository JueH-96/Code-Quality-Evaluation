class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 4:
            return 0
        
        # Precompute prefix counts
        prefix_counts = [{} for _ in range(n)]
        prefix_counts[0][nums[0]] = 1
        for i in range(1, n):
            prefix_counts[i] = prefix_counts[i-1].copy()
            num = nums[i]
            prefix_counts[i][num] = prefix_counts[i].get(num, 0) + 1
        
        # Precompute suffix counts
        suffix_counts = [{} for _ in range(n)]
        for r in range(n):
            if r + 2 >= n:
                suffix_counts[r] = {}
            else:
                current = {}
                for s in range(r + 2, n):
                    v = nums[s]
                    current[v] = current.get(v, 0) + 1
                suffix_counts[r] = current
        
        total = 0
        for q in range(n):
            for r in range(q + 2, n):
                denominator = nums[r]
                if denominator == 0:
                    continue
                current_suffix = suffix_counts[r]
                for v in current_suffix:
                    cnt_s = current_suffix[v]
                    numerator = nums[q] * v
                    if numerator % denominator != 0:
                        continue
                    X = numerator // denominator
                    if q - 2 < 0:
                        cnt_p = 0
                    else:
                        prev_counts = prefix_counts[q - 2]
                        cnt_p = prev_counts.get(X, 0)
                    total += cnt_p * cnt_s
        
        return total