class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        non_decreasing_count = 0
        
        # We use a two-pointer approach to find all subarrays and check if they can be made non-decreasing
        left = 0
        while left < n:
            right = left
            current_k = k
            while right < n:
                if right > left and nums[right] < nums[right - 1]:
                    needed = (nums[right - 1] - nums[right] + 1)
                    if current_k >= needed:
                        current_k -= needed
                        nums[right] += needed
                    else:
                        break
                right += 1
            
            # All subarrays starting from left and ending before right are non-decreasing
            non_decreasing_count += (right - left)
            
            # Reset the modified values for the next iteration
            nums[left:right] = nums[left:right]
            
            # Move to the next starting point
            left += 1
        
        return non_decreasing_count