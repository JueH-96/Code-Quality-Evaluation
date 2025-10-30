from collections import Counter
from typing import List

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        # 1. Count the frequency of each number in the input array.
        #    A Counter object is efficient for this.
        counts = Counter(nums)

        # 2. Iterate through the frequencies of all numbers.
        #    If any number appears more than twice, it's impossible to split
        #    the array as required, because you would have to place at least
        #    two instances of that number into the same sub-array (nums1 or nums2),
        #    violating the distinct element requirement.
        for count in counts.values():
            if count > 2:
                return False
        
        # 3. If the loop completes, it means no number appeared more than twice.
        #    In this scenario, it is always possible to form the two sub-arrays
        #    with distinct elements.
        #    - If a number appears once, put it in either nums1 or nums2.
        #    - If a number appears twice, put one instance in nums1 and the other in nums2.
        #    This guarantees distinctness in both arrays, and since the total
        #    length is even, and we distribute all numbers, the length requirement
        #    for nums1 and nums2 will also be met.
        return True