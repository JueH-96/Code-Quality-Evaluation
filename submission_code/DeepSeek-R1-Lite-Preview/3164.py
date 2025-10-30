from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        prev_count = 0
        for word in words:
            if word != "prev":
                nums.append(int(word))
                prev_count = 0
            else:
                prev_count += 1
                if prev_count <= len(nums):
                    result.append(nums[-prev_count])
                else:
                    result.append(-1)
        return result