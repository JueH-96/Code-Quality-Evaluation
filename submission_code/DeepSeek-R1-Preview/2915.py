from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        binary = [1 if (x % modulo == k) else 0 for x in nums]
        count_map = defaultdict(int)
        count_map[0] = 1  # Initial state with prefix 0 mod modulo
        current_sum = 0
        result = 0
        for num in binary:
            current_sum += num
            mod_current = current_sum % modulo
            required_mod = (mod_current - k) % modulo
            result += count_map.get(required_mod, 0)
            count_map[mod_current] += 1
        return result