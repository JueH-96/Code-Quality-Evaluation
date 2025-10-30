class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Frequency array for values from 0..n
        # (We include n because we'll check k = n as well,
        # but nums[i] will never be n due to 0 <= nums[i] < n.)
        freq = [0] * (n + 1)
        
        # Build frequency of each value in nums
        for val in nums:
            freq[val] += 1
        
        # Prefix array where prefix[k] = number of elements < k
        prefix = [0] * (n + 1)
        for k in range(1, n + 1):
            prefix[k] = prefix[k-1] + freq[k-1]
        
        # Count valid ways
        ways = 0
        for k in range(n + 1):
            # Condition 1: exactly k elements are < k
            if prefix[k] == k:
                # Condition 2: no element equals k
                if freq[k] == 0:
                    ways += 1
        
        return ways