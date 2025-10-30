class Solution:
    def minDifference(self, nums: List[int]) -> int:
        non_missing = [(i, val) for i, val in enumerate(nums) if val != -1]
        if not non_missing:
            return 0
        
        max_existing = 0
        for i in range(1, len(non_missing)):
            if non_missing[i][0] == non_missing[i-1][0] + 1:
                diff = abs(non_missing[i][1] - non_missing[i-1][1])
                if diff > max_existing:
                    max_existing = diff
        
        max_gap = 0
        for i in range(1, len(non_missing)):
            a_index, a_val = non_missing[i-1]
            b_index, b_val = non_missing[i]
            if b_index == a_index + 1:
                continue
            k = b_index - a_index - 1
            diff = abs(b_val - a_val)
            required_D = (diff + k) // (k + 1)
            if required_D > max_gap:
                max_gap = required_D
        
        return max(max_existing, max_gap)