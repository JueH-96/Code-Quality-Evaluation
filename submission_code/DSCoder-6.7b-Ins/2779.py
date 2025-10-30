class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0]*n
        ans = [0]*len(queries)
        cnt = 0
        for i in range(len(queries)):
            index, color = queries[i]
            if i > 0:
                ans[i] = ans[i-1]
            if index > 0 and nums[index-1] == color and nums[index-1] != 0:
                cnt -= 1
            if index < n-1 and nums[index+1] == color and nums[index+1] != 0:
                cnt -= 1
            nums[index] = color
            if index > 0 and nums[index-1] == color:
                cnt += 1
            if index < n-1 and nums[index+1] == color:
                cnt += 1
            ans[i] += cnt
        return ans