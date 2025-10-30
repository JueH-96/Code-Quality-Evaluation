class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        
        # Calculate prefix sum where each element is 1 if nums[i] % modulo == k
        for i in range(n):
            prefix[i + 1] = prefix[i] + (1 if nums[i] % modulo == k else 0)
            
        # Count frequency of remainders
        count = defaultdict(int)
        count[0] = 1
        result = 0
        
        # For each prefix sum, find number of valid subarrays ending at current position
        for i in range(1, n + 1):
            # Current prefix sum modulo
            curr = prefix[i] % modulo
            
            # Find target remainder that would make subarray interesting
            target = (curr - k) % modulo
            
            # Add count of subarrays with target remainder
            result += count[target]
            
            # Update count of current remainder
            count[curr] += 1
            
        return result