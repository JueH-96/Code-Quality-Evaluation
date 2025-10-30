class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        index1 = -1
        indexn = -1
        for i in range(n):
            if nums[i] == 1:
                index1 = i
            if nums[i] == n:
                indexn = i
        
        ans = 0
        if index1 > indexn:
            ans += index1
            ans += (n - 1 - indexn)
            ans -= 1

        else:
            ans += index1
            ans += (n - 1- indexn)


        return ans