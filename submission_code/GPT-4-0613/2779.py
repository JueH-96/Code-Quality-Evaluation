class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0]*n
        answer = []
        for index, color in queries:
            old_color = nums[index]
            nums[index] = color
            same_color_count = 0
            if index > 0 and nums[index-1] == color:
                same_color_count += 1
                if nums[index-1] == old_color:
                    same_color_count -= 1
            if index < n-1 and nums[index+1] == color:
                same_color_count += 1
                if nums[index+1] == old_color:
                    same_color_count -= 1
            if len(answer) > 0:
                answer.append(answer[-1] + same_color_count)
            else:
                answer.append(same_color_count)
        return answer