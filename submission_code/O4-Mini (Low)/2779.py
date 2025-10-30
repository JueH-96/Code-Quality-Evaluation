from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        ans = []
        same_adj = 0

        for idx, color in queries:
            old = nums[idx]
            # If the color is not changing, the count remains the same.
            if old == color:
                ans.append(same_adj)
                continue

            # Check left neighbor
            if idx > 0:
                if nums[idx - 1] == old and old != 0:
                    same_adj -= 1
                if nums[idx - 1] == color:
                    same_adj += 1

            # Check right neighbor
            if idx < n - 1:
                if nums[idx + 1] == old and old != 0:
                    same_adj -= 1
                if nums[idx + 1] == color:
                    same_adj += 1

            nums[idx] = color
            ans.append(same_adj)

        return ans