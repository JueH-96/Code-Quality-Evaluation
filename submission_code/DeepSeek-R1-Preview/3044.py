class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        reversed_nums = nums[::-1]
        collected = set()
        max_pos = -1
        for i, num in enumerate(reversed_nums):
            if 1 <= num <= k:
                if num not in collected:
                    collected.add(num)
                    if i > max_pos:
                        max_pos = i
            if len(collected) == k:
                break
        return max_pos + 1