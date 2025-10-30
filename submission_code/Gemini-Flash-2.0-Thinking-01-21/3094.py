from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = Counter(nums)
        total_operations = 0
        for count in counts.values():
            if count == 1:
                return -1
            else:
                operations_for_count = (count + (3 - count % 3) % 3) // 3
                total_operations += operations_for_count
        return total_operations