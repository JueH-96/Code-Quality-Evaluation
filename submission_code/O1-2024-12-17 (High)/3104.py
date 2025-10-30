class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        # Frequency array for all possible values 0..n (since nums[i] < n, freq[n] will stay 0)
        freq = [0] * (n + 1)
        for x in nums:
            freq[x] += 1
        
        # Build prefix sums so that prefix[k] = number of elements strictly less than k
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + freq[i]
        
        ways = 0
        # Check each possible size k of chosen set (from 0 to n)
        for k in range(n + 1):
            # Condition 1: exactly k elements of nums are < k
            if prefix[k] == k:
                # Condition 2: no element equals k (only relevant if k < n, otherwise freq[k] doesn't exist)
                if k < n:
                    if freq[k] == 0:
                        ways += 1
                else:
                    # When k == n, freq[n] is effectively 0 since nums[i] < n
                    ways += 1
        
        return ways