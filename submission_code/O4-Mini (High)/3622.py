from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        # Edge case
        if not nums:
            return 0
        
        # Determine the range of values in nums
        min_val = min(nums)
        max_val = max(nums)
        
        # Build frequency array for values [0..max_val]
        freq = [0] * (max_val + 1)
        for x in nums:
            freq[x] += 1
        
        # Build prefix‐sum array ps where ps[i] = sum of freq[0..i-1]
        ps = [0] * (max_val + 2)
        for i in range(max_val + 1):
            ps[i + 1] = ps[i] + freq[i]
        
        ans = 0
        # Only targets t in [min_val..max_val] need to be considered
        for t in range(min_val, max_val + 1):
            # Window [t-k .. t+k] clipped to [0..max_val]
            lw = t - k
            if lw < 0:
                lw = 0
            rw = t + k
            if rw > max_val:
                rw = max_val
            
            # Number of elements in nums within [t-k..t+k]
            window_sum = ps[rw + 1] - ps[lw] if lw <= rw else 0
            
            # eq = already equal to t (no op needed)
            eq = freq[t]
            # modable = those we can adjust to t (cost 1 op each)
            modable = window_sum - eq
            
            # We can only adjust up to numOperations elements
            count = eq + min(numOperations, modable)
            if count > ans:
                ans = count
        
        return ans