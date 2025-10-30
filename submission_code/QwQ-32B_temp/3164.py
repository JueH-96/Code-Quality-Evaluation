from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        consecutive_prev = 0
        
        for word in words:
            if word == "prev":
                consecutive_prev += 1
                k = consecutive_prev
                if k > len(nums):
                    result.append(-1)
                else:
                    result.append(nums[-k])
            else:
                nums.append(int(word))
                consecutive_prev = 0  # reset the consecutive counter
        return result