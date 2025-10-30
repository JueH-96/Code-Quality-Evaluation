import collections
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = collections.Counter(nums)
        ans = 1
        
        # Handle the case for x=1
        cnt1 = freq.get(1, 0)
        if cnt1 > 0:
            len_one = cnt1 if cnt1 % 2 == 1 else cnt1 - 1
            ans = max(ans, len_one)
        
        # Now handle x > 1 with freq[x] >= 2
        for x in freq:
            if x > 1 and freq[x] >= 2:
                # Try p from 4 to 1
                for p in range(4, 0, -1):
                    satisfied = True
                    # Check for i from 0 to p-1
                    for i in range(p):
                        val = x ** (1 << i)  # x ** (2**i)
                        if freq.get(val, 0) < 2:
                            satisfied = False
                            break
                    if not satisfied:
                        continue  # Not satisfied, try smaller p
                    # Check the p-th value
                    val_p = x ** (1 << p)  # x ** (2**p)
                    if freq.get(val_p, 0) >= 1:
                        length = 1 + 2 * p
                        ans = max(ans, length)
                        break  # No need to check smaller p for this x
        
        return ans