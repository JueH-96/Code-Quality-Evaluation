class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        count = 0
        
        for index, color in queries:
            if index > 0 and nums[index - 1] == color and nums[index - 1] != 0:
                count -= 1
            if index < n - 1 and nums[index + 1] == color and nums[index + 1] != 0:
                count -= 1
            if nums[index] != 0:
                if index > 0 and nums[index - 1] == nums[index]:
                    count -= 1
                if index < n - 1 and nums[index + 1] == nums[index]:
                    count -= 1
            nums[index] = color
            if index > 0 and nums[index - 1] == color and nums[index - 1] != 0:
                count += 1
            if index < n - 1 and nums[index + 1] == color and nums[index + 1] != 0:
                count += 1
            answer.append(count)
        
        return answer