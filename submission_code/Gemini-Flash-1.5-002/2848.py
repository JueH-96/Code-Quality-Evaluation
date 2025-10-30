class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        count = 0
        
        def is_special(perm):
            for i in range(n - 1):
                if perm[i] % perm[i+1] != 0 and perm[i+1] % perm[i] != 0:
                    return False
            return True

        import itertools
        for perm in itertools.permutations(nums):
            if is_special(list(perm)):
                count += 1
        return count % mod