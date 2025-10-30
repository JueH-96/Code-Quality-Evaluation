class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        current_count = 0
        answer = []
        for query in queries:
            index, color = query
            old_color = nums[index]
            new_color = color
            delta = 0
            if index > 0:
                before_left_good = (nums[index - 1] != 0 and old_color != 0 and nums[index - 1] == old_color)
                after_left_good = (nums[index - 1] != 0 and new_color != 0 and nums[index - 1] == new_color)
                if after_left_good and not before_left_good:
                    delta += 1
                elif before_left_good and not after_left_good:
                    delta -= 1
            if index < n - 1:
                before_right_good = (old_color != 0 and nums[index + 1] != 0 and old_color == nums[index + 1])
                after_right_good = (new_color != 0 and nums[index + 1] != 0 and new_color == nums[index + 1])
                if after_right_good and not before_right_good:
                    delta += 1
                elif before_right_good and not after_right_good:
                    delta -= 1
            current_count += delta
            nums[index] = new_color
            answer.append(current_count)
        return answer