class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        invalid_subarrays = 0
        
        # For each starting position
        for i in range(n):
            operations_needed = 0
            max_so_far = nums[i]
            
            # For each ending position
            for j in range(i + 1, n):
                # If current element is less than max_so_far,
                # we need operations to make it equal to max_so_far
                if nums[j] < max_so_far:
                    operations_needed += max_so_far - nums[j]
                    # If operations needed exceed k, this and all longer
                    # subarrays starting at i will be invalid
                    if operations_needed > k:
                        invalid_subarrays += n - j
                        break
                # Update max_so_far for next comparison
                max_so_far = max(max_so_far, nums[j])
        
        return total_subarrays - invalid_subarrays