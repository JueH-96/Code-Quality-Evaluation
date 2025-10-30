class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0]*n
        res = []
        color_count = [0]*n
        same_color_count = [0]*n
        for i in range(1, n):
            if nums[i-1] == nums[i]:
                color_count[i] = color_count[i-1] + 1
                if nums[i-1] != 0:
                    same_color_count[i] = same_color_count[i-1] + 1
            else:
                color_count[i] = color_count[i-1]
                same_color_count[i] = same_color_count[i-1]

        for q in queries:
            index, color = q[0], q[1]
            if nums[index] == 0:
                nums[index] = color
                if index > 0 and nums[index-1] == color:
                    color_count[index] += 1
                elif index > 0 and nums[index-1] != 0:
                    same_color_count[index] += 1
            elif nums[index] == color:
                if index > 0 and nums[index-1] == color:
                    color_count[index] += 1
                elif index > 0 and nums[index-1] != 0:
                    same_color_count[index] += 1
            else:
                if index > 0 and nums[index-1] == color:
                    color_count[index] = color_count[index-1] + 1
                elif index > 0 and nums[index-1] != 0:
                    same_color_count[index] = same_color_count[index-1] + 1
            res.append(color_count[index] + (0 if index > 0 and nums[index-1] == 0 else same_color_count[index]))
        return res