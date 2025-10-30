from collections import Counter
from typing import List

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        ans = 0
        for i in range(2, n - 2):
            left_counts = Counter(nums[:i])
            right_counts = Counter(nums[i + 1 :])
            for middle in set(nums):
                if nums[i] != middle:
                    continue
                left_ways = 0
                for left1 in left_counts:
                    for left2 in left_counts:
                        if left1 <= left2:
                            if left1 == left2:
                                left_ways += (
                                    left_counts[left1]
                                    * (left_counts[left1] - 1)
                                    // 2
                                )
                            else:
                                left_ways += left_counts[left1] * left_counts[left2]
                right_ways = 0
                for right1 in right_counts:
                    for right2 in right_counts:
                        if right1 <= right2:
                            if right1 == right2:
                                right_ways += (
                                    right_counts[right1]
                                    * (right_counts[right1] - 1)
                                    // 2
                                )
                            else:
                                right_ways += right_counts[right1] * right_counts[right2]
                if left_ways == 0 or right_ways == 0:
                    continue
                valid = True
                for x in set(nums):
                    if x == middle:
                        continue
                    count = (
                        (left_counts[x] if x in left_counts else 0)
                        + (right_counts[x] if x in right_counts else 0)
                        + (1 if x == nums[i] else 0)
                    )
                    if count >= (
                        (left_counts[middle] if middle in left_counts else 0)
                        + (right_counts[middle] if middle in right_counts else 0)
                        + 1
                    ):
                        valid = False
                        break
                if valid:
                    ans = (ans + left_ways * right_ways) % MOD
        return ans