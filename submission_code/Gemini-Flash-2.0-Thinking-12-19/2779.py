class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        current_count = 0
        for index, color in queries:
            old_color = nums[index]
            nums[index] = color

            # Check left neighbor
            if index > 0:
                if old_color != 0 and nums[index - 1] == old_color:
                    current_count -= 1
                if color != 0 and nums[index - 1] == color:
                    current_count += 1
                if color != old_color and nums[index-1] == old_color and old_color != 0 and nums[index-1] != 0:
                    pass # this case is already handled by the two conditions above

                if color != old_color and nums[index-1] == color and color != 0 and nums[index-1] != 0:
                    pass # this case is already handled by the two conditions above
                if color == old_color and color != 0 and nums[index-1] == color and nums[index-1] != 0:
                    if old_color == nums[index-1]:
                        pass
                    else:
                        pass


            # Check right neighbor
            if index < n - 1:
                if old_color != 0 and nums[index + 1] == old_color:
                    current_count -= 1
                if color != 0 and nums[index + 1] == color:
                    current_count += 1

            answer.append(current_count)
        return answer