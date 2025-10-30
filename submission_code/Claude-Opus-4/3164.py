class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []  # Store all integers seen so far
        result = []
        consecutive_prev = 0  # Count of consecutive "prev" strings
        
        for word in words:
            if word == "prev":
                consecutive_prev += 1
                # Check if k is within bounds
                if consecutive_prev <= len(nums):
                    # Get the (k-1)th element from the reversed nums
                    result.append(nums[-(consecutive_prev)])
                else:
                    result.append(-1)
            else:
                # It's an integer
                nums.append(int(word))
                consecutive_prev = 0  # Reset consecutive count
        
        return result