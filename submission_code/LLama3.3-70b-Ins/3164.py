from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        
        for word in words:
            if word == "prev":
                k = 1
                while words[words.index(word) - k] == "prev":
                    k += 1
                if k > len(nums):
                    result.append(-1)
                else:
                    result.append(nums[-k])
            else:
                nums.append(int(word))
        
        return result