class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        current_count = 0
        result = []
        for q in queries:
            i, c = q[0], q[1]
            # Subtract contributions from the current color before changing
            if i > 0:
                if nums[i-1] != 0 and nums[i] != 0 and nums[i-1] == nums[i]:
                    current_count -= 1
            if i < n - 1:
                if nums[i] != 0 and nums[i+1] != 0 and nums[i] == nums[i+1]:
                    current_count -= 1
            # Change the color
            nums[i] = c
            # Add contributions from the new color
            if i > 0:
                if nums[i-1] != 0 and nums[i] != 0 and nums[i-1] == nums[i]:
                    current_count += 1
            if i < n - 1:
                if nums[i] != 0 and nums[i+1] != 0 and nums[i] == nums[i+1]:
                    current_count += 1
            result.append(current_count)
        return result