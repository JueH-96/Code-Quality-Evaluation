from collections import deque

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        left = 0
        right = 0
        count = 0
        max_count = 0
        window = deque()
        
        while right < len(nums):
            if nums[right] == max_num:
                window.append(right)
                max_count += 1
                
                # if the window has the max_num k times, count the number of valid subarrays
                while max_count == k:
                    curr_left = window[0]
                    curr_right = window[-1]
                    count += (curr_right - curr_left) * (left + 1)
                    
                    # remove the leftmost occurrence of the max_num from the window
                    window.popleft()
                    max_count -= 1
                    left = curr_left + 1
                    
            right += 1
            
        return count