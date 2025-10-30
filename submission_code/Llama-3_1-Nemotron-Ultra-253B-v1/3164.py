from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        current_consecutive_prev = 0
        result = []
        for word in words:
            if word == "prev":
                current_consecutive_prev += 1
                k = current_consecutive_prev
                if k > len(nums):
                    result.append(-1)
                else:
                    reversed_nums = nums[::-1]
                    result.append(reversed_nums[k-1])
            else:
                nums.append(int(word))
                current_consecutive_prev = 0
        return result