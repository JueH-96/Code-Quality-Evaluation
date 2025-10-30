class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10 ** 9 + 7
        
        def backtrack(start, perm):
            if len(perm) == n:
                return 1
            
            count = 0
            for i in range(start, n):
                if (not perm or perm[-1] % nums[i] == 0) or (nums[i] % perm[-1] == 0):
                    perm.append(nums[i])
                    count = (count + backtrack(i + 1, perm)) % mod
                    perm.pop()
            
            return count
        
        return backtrack(0, [])