class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def lcm(a, b):
            return (a * b) // gcd(a, b)

        n = len(nums)
        max_len = 0

        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                prod = 1
                g = subarray[0]
                l = subarray[0]

                for num in subarray:
                    prod *= num
                    g = gcd(g, num)
                    l = lcm(l, num)

                if prod == l * g:
                    max_len = max(max_len, len(subarray))

        return max_len