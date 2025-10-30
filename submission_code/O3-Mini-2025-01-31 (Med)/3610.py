from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        res = []
        # Process each sliding window of length k.
        for i in range(n - k + 1):
            window = nums[i:i+k]
            cnt = Counter(window)
            # If the subarray has less than x distinct elements, use the sum of the entire subarray.
            if len(cnt) < x:
                res.append(sum(window))
            else:
                # Build a list of (element, frequency) pairs.
                # We sort by frequency descending, and then by element's value descending.
                freq_list = sorted(cnt.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)
                # Select the top x elements.
                top_x = set([freq_list[j][0] for j in range(x)])
                # Sum the occurrences of only the selected elements.
                s = sum(val for val in window if val in top_x)
                res.append(s)
        return res