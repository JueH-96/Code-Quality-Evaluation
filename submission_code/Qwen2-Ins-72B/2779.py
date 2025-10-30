class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        res = []
        count = 0
        for idx, color in queries:
            prev = nums[idx]
            nums[idx] = color
            for i in [idx - 1, idx + 1]:
                if 0 <= i < n:
                    if nums[i] == color and nums[i] != 0:
                        count += 1
                    elif nums[i] == prev and nums[i] != 0:
                        count -= 1
            res.append(count)
        return res