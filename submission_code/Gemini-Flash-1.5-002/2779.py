class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        ans = []
        for index, color in queries:
            nums[index] = color
            count = 0
            for i in range(n - 1):
                if nums[i] != 0 and nums[i] == nums[i+1]:
                    count += 1
            ans.append(count)
        return ans