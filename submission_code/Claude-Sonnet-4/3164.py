class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result = []
        nums = []  # Store all integers seen so far
        consecutive_prev_count = 0
        
        for word in words:
            if word == "prev":
                consecutive_prev_count += 1
                # Get the (consecutive_prev_count - 1)th index from reversed nums
                if consecutive_prev_count <= len(nums):
                    # nums_reverse[k-1] where k is consecutive_prev_count
                    nums_reverse = nums[::-1]
                    result.append(nums_reverse[consecutive_prev_count - 1])
                else:
                    result.append(-1)
            else:
                # It's an integer, reset consecutive prev count
                consecutive_prev_count = 0
                nums.append(int(word))
        
        return result