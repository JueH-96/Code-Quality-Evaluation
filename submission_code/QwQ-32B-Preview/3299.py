from typing import List
from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        max_size = 0

        for x in sorted(freq.keys()):
            if x == 1:
                count = freq[1]
                if count % 2 == 1:
                    size = count
                else:
                    size = count - 1
                if size > max_size:
                    max_size = size
            else:
                sequence = []
                current = x
                while current in freq:
                    sequence.append(current)
                    current = current * current
                m_max = len(sequence)
                for m in range(m_max + 1):
                    valid = True
                    for i in range(m):
                        if freq[sequence[i]] < 2:
                            valid = False
                            break
                    if m < m_max:
                        if freq[sequence[m]] < 1:
                            valid = False
                    else:
                        break
                    if valid:
                        current_size = 2 * m + 1
                        if current_size > max_size:
                            max_size = current_size
        return max_size