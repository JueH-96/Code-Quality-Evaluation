from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # ops[i] will record how much to subtract from curr when we reach index i
        ops = [0] * (n + 1)
        curr = 0  # current total of active subtractions at index i
        
        for i in range(n):
            # remove expired subtractions
            curr += ops[i]
            # how many more times we need to apply the k-window operation at i
            need = nums[i] - curr
            if need < 0:
                # we've subtracted too much
                return False
            if need > 0:
                # if we start a window here, it must fit in the array
                if i + k > n:
                    return False
                # add these need subtractions to curr, and schedule their end
                curr += need
                ops[i + k] -= need
        
        return True