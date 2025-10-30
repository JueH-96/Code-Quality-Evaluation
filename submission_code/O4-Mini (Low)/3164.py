from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        consecutive_prev = 0
        
        for w in words:
            if w == "prev":
                consecutive_prev += 1
                # if we have seen enough integers to go back `consecutive_prev` steps
                if consecutive_prev <= len(nums):
                    result.append(nums[-consecutive_prev])
                else:
                    result.append(-1)
            else:
                # it's an integer
                nums.append(int(w))
                consecutive_prev = 0
        
        return result