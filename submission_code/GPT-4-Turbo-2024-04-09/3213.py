class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Find the maximum element in the array
        max_element = max(nums)
        
        # Find all positions of the maximum element
        max_positions = [i for i, num in enumerate(nums) if num == max_element]
        
        # If there are fewer than k occurrences of the maximum element, no valid subarray exists
        if len(max_positions) < k:
            return 0
        
        # Total number of subarrays
        total_subarrays = 0
        
        # We need to consider subarrays that include the maximum element at least k times
        # We will use the positions of the maximum element to determine valid subarrays
        n = len(nums)
        
        # We will iterate over each group of k consecutive positions of the max element
        for i in range(len(max_positions) - k + 1):
            # Start of the subarray that includes at least k max elements
            start = max_positions[i]
            
            # End of the subarray that includes at least k max elements
            end = max_positions[i + k - 1]
            
            # To count all subarrays that start before the first of these k elements and end after the last
            # We calculate the number of options to choose the start and end of the subarray
            # left_count is the number of ways to choose the start of the subarray
            # right_count is the number of ways to choose the end of the subarray
            left_count = start + 1 if i == 0 else start - max_positions[i - 1]
            right_count = n - end if i + k == len(max_positions) else max_positions[i + k] - end
            
            # Multiply the counts to get the number of subarrays for this segment
            total_subarrays += left_count * right_count
        
        return total_subarrays