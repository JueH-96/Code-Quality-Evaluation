from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        prev_count = 0
        
        for word in words:
            if word == "prev":
                prev_count += 1
                if prev_count <= len(nums):
                    result.append(nums[-prev_count])
                else:
                    result.append(-1)
            else:
                nums.append(int(word))
                prev_count = 0  # Reset prev_count when a number is encountered
        
        return result