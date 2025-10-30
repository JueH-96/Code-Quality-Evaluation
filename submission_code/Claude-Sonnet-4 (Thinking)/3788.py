class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        max_sum = float('-inf')
        
        # Try all possible ways to keep elements (at least one must remain)
        for mask in range(1, 1 << n):
            remaining = []
            for i in range(n):
                if mask & (1 << i):
                    remaining.append(nums[i])
            
            # Find max sum contiguous subarray with unique elements
            curr_max = self.maxUniqueSubarray(remaining)
            max_sum = max(max_sum, curr_max)
        
        return max_sum
    
    def maxUniqueSubarray(self, arr):
        n = len(arr)
        max_sum = float('-inf')
        
        # Try all contiguous subarrays and find max sum with unique elements
        for i in range(n):
            seen = set()
            current_sum = 0
            for j in range(i, n):
                if arr[j] in seen:
                    break
                seen.add(arr[j])
                current_sum += arr[j]
                max_sum = max(max_sum, current_sum)
        
        return max_sum