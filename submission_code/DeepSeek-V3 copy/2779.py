class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        count = 0
        
        for idx, color in queries:
            # Check the left neighbor
            if idx > 0:
                if nums[idx] != 0 and nums[idx] == nums[idx - 1]:
                    count -= 1
                if color != 0 and color == nums[idx - 1]:
                    count += 1
            # Check the right neighbor
            if idx < n - 1:
                if nums[idx] != 0 and nums[idx] == nums[idx + 1]:
                    count -= 1
                if color != 0 and color == nums[idx + 1]:
                    count += 1
            # Update the color
            nums[idx] = color
            answer.append(count)
        
        return answer