from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        total = 0
        answer = []
        for i, c in queries:
            prev = nums[i]
            if prev != 0:
                # Subtract contributions from left and right neighbors
                for neighbor in [i - 1, i + 1]:
                    if 0 <= neighbor < n and nums[neighbor] == prev:
                        total -= 1
            # Update the color
            nums[i] = c
            # Add contributions from new color
            for neighbor in [i - 1, i + 1]:
                if 0 <= neighbor < n and nums[neighbor] == c:
                    total += 1
            answer.append(total)
        return answer