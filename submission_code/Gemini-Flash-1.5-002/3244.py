class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        
        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)

        if not nums:
            return 0

        g = nums[0]
        for i in range(1, len(nums)):
            g = gcd(g, nums[i])

        if g == 1:
            return 1
        else:
            count = 0
            for num in nums:
                if num % g !=0:
                    count +=1
            if count == 0:
                return 1
            else:
                return count