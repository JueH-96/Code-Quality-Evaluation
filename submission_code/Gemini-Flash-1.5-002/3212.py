class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        count = {}
        dp = {}

        def solve(i):
            if i == n:
                return 1
            if i in dp:
                return dp[i]
            
            ans = 0
            seen = set()
            for j in range(i, n):
                seen_in_subarray = set()
                valid = True
                for k in range(i, j + 1):
                    if nums[k] in seen:
                        valid = False
                        break
                    if nums[k] in count:
                        valid = False
                        break
                    seen_in_subarray.add(nums[k])

                if valid:
                    temp_count = count.copy()
                    for num in seen_in_subarray:
                        count[num] = count.get(num,0) + 1
                    ans = (ans + solve(j + 1)) % MOD
                    count = temp_count

            dp[i] = ans
            return ans

        result = solve(0)
        return result