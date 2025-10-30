from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # A number ends with at least one binary 0  ⇔  it is even.
        # The OR of several numbers ends with a 0  ⇔  every one of them ends with a 0.
        # Therefore we only need to know whether we can choose at least two even
        # elements from the array.
        even_cnt = sum(1 for x in nums if x & 1 == 0)
        return even_cnt >= 2