from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result = []  # Stores the integers to be returned for "prev" operations
        visited_nums = [] # Stores all integers encountered so far, in order
        consecutive_prev_count = 0 # Counts consecutive "prev" strings

        for word in words:
            if word == "prev":
                consecutive_prev_count += 1
                k = consecutive_prev_count # k is the number of consecutive "prev" strings

                # Check if k is greater than the total number of integers visited
                # If true, append -1 as per the problem statement.
                if k > len(visited_nums):
                    result.append(-1)
                else:
                    # The problem asks for the (k-1)-th index of nums_reverse.
                    # This is equivalent to the k-th element from the end of the original nums array.
                    # For a 0-indexed list `visited_nums` of length `L`,
                    # the k-th element from the end is at index `L - k`.
                    result.append(visited_nums[len(visited_nums) - k])
            else:
                # If the word is an integer string
                num = int(word)
                visited_nums.append(num)
                # Reset consecutive_prev_count if a non-"prev" word is encountered,
                # as the sequence of "prev" strings is broken.
                consecutive_prev_count = 0
        
        return result