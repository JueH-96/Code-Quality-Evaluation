class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        n = len(nums)
        
        for i in range(n - 1):
            # Check if we can start an alternating subarray at position i
            if nums[i + 1] == nums[i] + 1:
                # Found a potential start
                length = 2
                j = i + 1
                
                # Keep extending while the alternating pattern continues
                while j + 1 < n:
                    # Check if the pattern continues
                    if (j - i) % 2 == 0:  # Even position relative to start
                        if nums[j + 1] == nums[i] + 1:
                            length += 1
                            j += 1
                        else:
                            break
                    else:  # Odd position relative to start
                        if nums[j + 1] == nums[i]:
                            length += 1
                            j += 1
                        else:
                            break
                
                max_length = max(max_length, length)
        
        return max_length