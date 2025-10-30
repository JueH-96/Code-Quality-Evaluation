class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n  # Track the current color at each index
        adj_same = 0    # Current number of adjacent pairs with the same color
        ans = []

        for idx, color in queries:
            old_color = nums[idx]
            # If the color is the same, no change
            if old_color == color:
                ans.append(adj_same)
                continue

            # Decrease count for old_color neighbors if they matched
            if idx - 1 >= 0 and nums[idx - 1] == old_color and old_color != 0:
                adj_same -= 1
            if idx + 1 < n and nums[idx + 1] == old_color and old_color != 0:
                adj_same -= 1

            # Update the color
            nums[idx] = color

            # Increase count for new_color neighbors if they match
            if idx - 1 >= 0 and nums[idx - 1] == color and color != 0:
                adj_same += 1
            if idx + 1 < n and nums[idx + 1] == color and color != 0:
                adj_same += 1

            ans.append(adj_same)

        return ans