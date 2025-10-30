class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        answer = 0
        current_length = 0
        last_parity = -1  # Sentinel value to indicate not in a subarray yet
        
        for num in nums:
            if current_length == 0:
                # Try to start a new subarray if num is even and <= threshold
                if num % 2 == 0 and num <= threshold:
                    current_length = 1
                    last_parity = num % 2
                else:
                    current_length = 0
                    last_parity = -1
            else:
                # We are in a subarray, check if we can extend it
                if num <= threshold and (num % 2 != last_parity):
                    current_length += 1
                    last_parity = num % 2
                else:
                    # Can't continue; try to start a new subarray from current num
                    if num % 2 == 0 and num <= threshold:
                        current_length = 1
                        last_parity = 0
                    else:
                        current_length = 0
                        last_parity = -1
            
            answer = max(answer, current_length)
        
        return answer