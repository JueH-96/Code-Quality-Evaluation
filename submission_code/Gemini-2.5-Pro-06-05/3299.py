import collections
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        """
        Calculates the maximum length of a subset of nums that can form a sequence
        of the form [x, x^2, x^4, ..., x^(2^p), ..., x^4, x^2, x].
        """
        counts = collections.Counter(nums)
        
        max_len = 0
        
        # Handle the special case where x = 1.
        # The sequence is [1, 1, ..., 1]. The length must be odd.
        if 1 in counts:
            c1 = counts.pop(1)
            if c1 % 2 == 0:
                max_len = c1 - 1
            else:
                max_len = c1
        
        # If there are any numbers other than 1, the answer is at least 1,
        # as any single number forms a valid sequence of length 1.
        if counts:
            max_len = max(max_len, 1)

        # Iterate through potential starting numbers k > 1.
        # Sorting allows us to start chains from their smallest elements.
        sorted_keys = sorted(counts.keys())
        
        for k in sorted_keys:
            # If k was already "used" as part of a chain started by a smaller number,
            # it would have been popped from counts.
            if k not in counts:
                continue
            
            length = 0
            curr = k
            while curr in counts:
                count = counts.pop(curr)
                
                if count >= 2:
                    length += 2
                    # The next term in the sequence is curr*curr.
                    # This will quickly exceed the maximum possible value in nums.
                    # The `curr in counts` check implicitly handles this.
                    curr = curr * curr
                else:  # count == 1
                    # This number must be the peak.
                    length += 1
                    max_len = max(max_len, length)
                    break
            else:  # The while loop completed without a `break`.
                # This means the last number `c` in the chain had a count >= 2,
                # but its square was not in nums. So, `c` must be the peak.
                # The length was calculated assuming `c` was a pair, so we correct by subtracting 1.
                if length > 0:
                    max_len = max(max_len, length - 1)
                    
        return max_len