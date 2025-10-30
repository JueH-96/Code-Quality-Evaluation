from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = Counter(nums)
        max_size = 0

        # Handle the special case for x=1
        if 1 in count:
            cnt = count[1]
            if cnt % 2 == 1:
                temp = cnt
            else:
                temp = cnt - 1
            if temp > 0:
                max_size = temp

        # Process all other numbers
        for x in count:
            if x == 1:
                continue

            # Generate the chain
            chain = []
            current = x
            while True:
                if current not in count:
                    break
                chain.append(current)
                next_val = current * current
                # Prevent infinite loops for cycles (though unlikely for x != 1)
                if next_val in chain:
                    break
                current = next_val

            # Check all possible d in reversed order
            found = False
            for d in reversed(range(len(chain))):
                valid = True
                # Check all elements in 0..d-1 have >=2 counts
                for i in range(d):
                    if count[chain[i]] < 2:
                        valid = False
                        break
                if valid and (count[chain[d]] >= 1):
                    current_size = 2 * d + 1
                    if current_size > max_size:
                        max_size = current_size
                    found = True
                    break
            # If no valid d found, still consider d=0 for subset size 1
            if not found and len(chain) > 0:
                # Check d=0 case explicitly
                if count[chain[0]] >= 1:
                    if 1 > max_size:
                        max_size = 1

        return max_size if max_size != 0 else 1