class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0

        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def lcm(a, b):
            return (a * b) // gcd(a, b)

        def get_gcd(arr):
            res = arr[0]
            for i in range(1, len(arr)):
                res = gcd(res, arr[i])
            return res

        def get_lcm(arr):
            res = arr[0]
            for i in range(1, len(arr)):
                res = lcm(res, arr[i])
            return res

        for i in range(n + 1):
            temp = []
            if i == n:
                if not nums:
                    score = 0
                else:
                    g = get_gcd(nums)
                    l = get_lcm(nums)
                    score = g * l
                ans = max(ans, score)

            else:
                temp = nums[:i] + nums[i+1:]
                if not temp:
                    score = 0
                else:
                    g = get_gcd(temp)
                    l = get_lcm(temp)
                    score = g * l
                ans = max(ans, score)
        
        return ans