class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        max_sum = float('-inf')
        
        # Try all possible subarrays
        for i in range(n):
            seen = set()  # To track unique elements
            curr_sum = 0
            
            # For each starting point i, try extending the subarray
            for j in range(i, n):
                # If current element is already seen, skip this subarray
                if nums[j] in seen:
                    continue
                    
                # Add current element to seen set and sum
                seen.add(nums[j])
                curr_sum += nums[j]
                
                # Update max_sum if current sum is larger
                max_sum = max(max_sum, curr_sum)
        
        return max_sum