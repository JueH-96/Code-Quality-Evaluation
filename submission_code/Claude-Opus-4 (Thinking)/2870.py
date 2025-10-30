class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = -1
        
        for i in range(n - 1):
            # Check if we can start an alternating subarray at position i
            if nums[i + 1] == nums[i] + 1:
                # We have a valid start with nums[i] and nums[i+1]
                first_val = nums[i]
                second_val = nums[i + 1]
                length = 2
                j = i + 2
                
                # Extend as much as possible
                while j < n:
                    expected = first_val if (j - i) % 2 == 0 else second_val
                    if nums[j] == expected:
                        length += 1
                        j += 1
                    else:
                        break
                
                max_length = max(max_length, length)
        
        return max_length