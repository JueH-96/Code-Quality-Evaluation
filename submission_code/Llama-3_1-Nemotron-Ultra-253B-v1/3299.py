from collections import defaultdict
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        max_len = 0
        for x in freq:
            current_k = 0
            current_square = x
            while True:
                next_square = current_square * current_square
                if freq[current_square] >= 2 and next_square in freq and freq[next_square] >= 1:
                    current_k += 1
                    current_square = next_square
                else:
                    break
            max_len = max(max_len, 2 * current_k + 1)
        return max_len