from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        count = 0

        for word in reversed(words):
            if word == "prev":
                count += 1
                if count <= len(nums):
                    result.append(nums[-count])
                else:
                    result.append(-1)
            else:
                count = 0
                nums.append(int(word))

        return result[::-1]