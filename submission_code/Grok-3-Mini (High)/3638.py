from typing import *
import math
from collections import Counter

class Solution:
    def makeStringGood(self, s: str) -> int:
        freq = Counter(s)
        freq_values = list(freq.values())
        if not freq_values:
            return 0  # Not necessary given constraints, but safe
        max_freq = max(freq_values)
        min_ops = float('inf')
        for K in range(0, max_freq + 1):
            current_sum = 0
            for f in freq_values:
                current_sum += min(abs(f - K), f)
            if current_sum < min_ops:
                min_ops = current_sum
        return min_ops