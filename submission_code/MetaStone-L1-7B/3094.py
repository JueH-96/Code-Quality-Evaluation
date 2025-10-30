from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        freq = Counter(nums)
        for count in freq.values():
            if count < 2:
                return -1
        total = 0
        for count in freq.values():
            b_max = count // 3
            for b in range(b_max, -1, -1):
                rem = count - 3 * b
                if rem % 2 == 0 and rem >= 0:
                    a = rem // 2
                    total += (a + b)
                    break
        return total