from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []                    # all integers encountered so far
        consecutive_prev = 0         # length of current consecutive "prev" run
        ans = []                     # answers for every "prev"
        
        for w in words:
            if w == "prev":
                consecutive_prev += 1
                # Determine the (k-1)-th element from the end where k = consecutive_prev
                if consecutive_prev <= len(nums):
                    ans.append(nums[-consecutive_prev])
                else:
                    ans.append(-1)
            else:                    # w is an integer
                nums.append(int(w))
                consecutive_prev = 0 # reset the run length
        return ans