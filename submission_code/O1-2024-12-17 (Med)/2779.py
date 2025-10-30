class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n  # initially all 0 (uncolored)
        adj_sum = 0     # keep track of the count of adjacent equal (non-zero) pairs
        answer = []

        for idx, new_color in queries:
            old_color = nums[idx]
            # If the color is not changing, no need to update adjacencies
            if old_color == new_color:
                answer.append(adj_sum)
                continue

            # If there was an old color, remove its contributions from adj_sum
            if old_color != 0:
                if idx > 0 and nums[idx - 1] == old_color:
                    adj_sum -= 1
                if idx < n - 1 and nums[idx + 1] == old_color:
                    adj_sum -= 1

            # Assign the new color
            nums[idx] = new_color

            # Add new contributions with the new color
            if idx > 0 and nums[idx - 1] == new_color:
                adj_sum += 1
            if idx < n - 1 and nums[idx + 1] == new_color:
                adj_sum += 1

            # Record the current adj_sum
            answer.append(adj_sum)

        return answer