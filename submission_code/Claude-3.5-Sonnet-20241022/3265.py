class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')
        found = False
        
        # For each possible starting index
        for i in range(n):
            # For each possible ending index
            for j in range(i, n):
                # Check if absolute difference between first and last element is k
                if abs(nums[i] - nums[j]) == k:
                    found = True
                    # Calculate sum of subarray
                    curr_sum = sum(nums[i:j+1])
                    max_sum = max(max_sum, curr_sum)
        
        return max_sum if found else 0