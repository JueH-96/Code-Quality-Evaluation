from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        current_count = 0

        for index, color in queries:
            # Update the color at the given index
            old_color = nums[index]
            nums[index] = color

            # Check the adjacent elements and update the count
            if index > 0 and nums[index] == nums[index - 1] and nums[index] != 0:
                current_count += 1
            if index < n - 1 and nums[index] == nums[index + 1] and nums[index] != 0:
                current_count += 1
            if index > 0 and old_color == nums[index - 1] and old_color != 0:
                current_count -= 1
            if index < n - 1 and old_color == nums[index + 1] and old_color != 0:
                current_count -= 1

            answer.append(current_count)

        return answer