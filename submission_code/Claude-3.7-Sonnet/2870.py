class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = -1  # Initialize to -1 in case no valid subarray exists
        
        for i in range(n-1):  # Try each possible starting position
            first = nums[i]
            second = nums[i+1]
            
            # Check if the second element is one more than the first
            if second == first + 1:
                length = 2  # We have at least a valid subarray of length 2
                
                # Try to extend the alternating pattern
                for j in range(i+2, n):
                    # Expected value alternates between first and second
                    expected = first if (j - i) % 2 == 0 else second
                    if nums[j] == expected:
                        length += 1
                    else:
                        break
                
                max_length = max(max_length, length)
        
        return max_length