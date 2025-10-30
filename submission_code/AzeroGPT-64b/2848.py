class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
    
        # Find all numbers that can divide each number in nums, considering gcd is possible largest number.
        gcd_tuples = [set() for _ in nums]
        for i, num in enumerate(nums):
            for j, other_num in enumerate(nums):
                if i != j and (num % other_num == 0 or other_num % num == 0):
                    gcd_tuples[i].add(j)
        
        @cache
        def dfs(index, used_mask):
            if used_mask == (1 << n) - 1: return 1
            res = 0
            for next_index in gcd_tuples[index]:
                if not used_mask & (1 << next_index):
                    res += dfs(next_index, used_mask | (1 << next_index))
            return res % MOD
        
        total_permutations = sum(dfs(index, 1 << index) for index in range(n)) % MOD
        return total_permutations