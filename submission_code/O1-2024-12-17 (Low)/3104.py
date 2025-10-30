class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Frequency array to count how many times each value appears
        freq = [0] * (n+1)  # We use n+1 so we can safely reference freq[n]
        for val in nums:
            freq[val] += 1
        
        # Prefix sums p[k] = number of elements < k
        p = [0] * (n+1)
        for k in range(1, n+1):
            p[k] = p[k-1] + freq[k-1]
        
        # Count valid k's (subset sizes)
        # Condition: p[k] == k  (means exactly k elements are < k) AND freq[k] == 0 (no element equals k)
        count = 0
        for k in range(n+1):
            if p[k] == k and freq[k] == 0:
                count += 1
        
        return count