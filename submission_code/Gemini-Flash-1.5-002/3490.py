class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n
        
        max_len = 0
        for i in range(1 << n):
            subsequence = []
            for j in range(n):
                if (i >> j) & 1:
                    subsequence.append(nums[j])
            
            if len(subsequence) < 2:
                max_len = max(max_len, len(subsequence))
                continue

            is_valid = True
            first_sum_parity = (subsequence[0] + subsequence[1]) % 2
            for k in range(1, len(subsequence) -1):
                if (subsequence[k] + subsequence[k+1]) % 2 != first_sum_parity:
                    is_valid = False
                    break
            if is_valid:
                max_len = max(max_len, len(subsequence))

        return max_len