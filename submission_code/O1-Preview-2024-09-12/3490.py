class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        parities = [num % 2 for num in nums]
        count_zeros = parities.count(0)
        count_ones = len(parities) - count_zeros
        L0 = max(count_zeros, count_ones)
        
        # For sequence starting with 0
        lcs0 = 0
        expected0 = 0
        # For sequence starting with 1
        lcs1 = 0
        expected1 = 1
        
        for parity in parities:
            if parity == expected0:
                lcs0 += 1
                expected0 = 1 - expected0
            if parity == expected1:
                lcs1 += 1
                expected1 = 1 - expected1
        
        L1 = max(lcs0, lcs1)
        return max(L0, L1)