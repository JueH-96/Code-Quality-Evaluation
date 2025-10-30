class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = -1
        
        for i in range(n-1):
            if nums[i+1] - nums[i] == 1:
                curr_len = 2
                j = i + 1
                
                while j < n-1:
                    if nums[j+1] - nums[j] == -1 and nums[j+1] == nums[i]:
                        curr_len += 1
                        j += 1
                        if j < n-1 and nums[j+1] - nums[j] == 1:
                            curr_len += 1
                            j += 1
                        else:
                            break
                    else:
                        break
                        
                max_len = max(max_len, curr_len)
                
        return max_len