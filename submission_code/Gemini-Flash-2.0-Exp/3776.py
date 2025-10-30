class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n % 2 == 1:
            nums.append(0)
            n += 1

        dp = {}

        def solve(arr):
            t = tuple(arr)
            if t in dp:
                return dp[t]

            if not arr:
                return 0

            if len(arr) <= 2:
                if not arr:
                    return 0
                else:
                    cost = max(arr) if arr else 0
                    dp[t] = cost
                    return cost
            
            ans = float('inf')
            
            # Remove 0 and 1
            cost1 = max(arr[0], arr[1])
            ans = min(ans, cost1 + solve(arr[2:]))
            
            # Remove 0 and 2
            cost2 = max(arr[0], arr[2])
            ans = min(ans, cost2 + solve(arr[1:]))
            
            # Remove 1 and 2
            cost3 = max(arr[1], arr[2])
            ans = min(ans, cost3 + solve([arr[0]] + arr[3:]))
            
            dp[t] = ans
            return ans

        return solve(nums)