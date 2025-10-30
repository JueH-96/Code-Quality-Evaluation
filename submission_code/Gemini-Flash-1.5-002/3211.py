class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = {}
        def solve(i, prev):
            if i == n:
                return 0
            if (i, prev) in dp:
                return dp[(i, prev)]
            
            ans = solve(i + 1, prev) #skip current element

            curr_sum = prev
            for j in range(i,n):
                curr_sum += nums[j]
                if curr_sum >= prev:
                    ans = max(ans, 1 + solve(j + 1, curr_sum))
                else:
                    break
            dp[(i,prev)] = ans
            return ans

        return solve(0,0)