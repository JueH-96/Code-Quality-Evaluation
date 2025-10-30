from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Helper function to count the number of set bits in an integer
        def count_set_bits(n: int) -> int:
            return bin(n).count('1')

        # Group numbers by the number of set bits
        from collections import defaultdict
        groups = defaultdict(list)
        for num in nums:
            groups[count_set_bits(num)].append(num)

        # Check if each group is sorted
        for group in groups.values():
            if sorted(group) != group:
                return false

        return true