class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        count = 0
        for index_i, color_i in queries:
            old_color = nums[index_i]
            if index_i > 0 and nums[index_i - 1] != 0 and nums[index_i - 1] == old_color:
                count -= 1
            if index_i < n - 1 and nums[index_i + 1] != 0 and nums[index_i + 1] == old_color:
                count -= 1
            nums[index_i] = color_i
            if index_i > 0 and nums[index_i - 1] != 0 and nums[index_i - 1] == color_i:
                count += 1
            if index_i < n - 1 and nums[index_i + 1] != 0 and nums[index_i + 1] == color_i:
                count += 1
            answer.append(count)
        return answer