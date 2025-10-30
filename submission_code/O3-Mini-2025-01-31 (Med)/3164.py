from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result = []
        nums = []  # list of integers we've seen so far
        consecutive_prev = 0  # count consecutive "prev" strings
        
        for word in words:
            if word != "prev":
                # We got an integer; add it to our visited integer list
                nums.append(int(word))
                consecutive_prev = 0  # reset consecutive "prev" count
            else:
                # We encountered a "prev"
                consecutive_prev += 1  # increment consecutive count
                # Check if we have enough integers visited
                if consecutive_prev <= len(nums):
                    # The k-th last visited integer is at index -consecutive_prev
                    result.append(nums[-consecutive_prev])
                else:
                    result.append(-1)
                    
        return result