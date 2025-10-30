class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        current_count = 0
        ans = []
        for index, color in queries:
            prev_color = nums[index]
            if prev_color != 0:
                # Subtract contributions from left neighbor
                if index > 0 and nums[index - 1] == prev_color:
                    current_count -= 1
                # Subtract contributions from right neighbor
                if index < n - 1 and nums[index + 1] == prev_color:
                    current_count -= 1
            # Update the color
            nums[index] = color
            # Add contributions from new color to left neighbor
            if index > 0 and nums[index - 1] == color:
                current_count += 1
            # Add contributions from new color to right neighbor
            if index < n - 1 and nums[index + 1] == color:
                current_count += 1
            ans.append(current_count)
        return ans