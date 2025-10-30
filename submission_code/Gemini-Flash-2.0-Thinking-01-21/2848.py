class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        memo = {}
        MOD = 10**9 + 7
        
        def count_special_permutations(last_index, used_mask):
            if (last_index, used_mask) in memo:
                return memo[(last_index, used_mask)]
            if bin(used_mask).count('1') == n:
                return 1
            
            count = 0
            for j in range(n):
                if not (used_mask & (1 << j)):
                    valid_placement = False
                    if last_index == -1:
                        valid_placement = True
                    else:
                        if nums[last_index] % nums[j] == 0 or nums[j] % nums[last_index] == 0:
                            valid_placement = True
                            
                    if valid_placement:
                        next_used_mask = used_mask | (1 << j)
                        count = (count + count_special_permutations(j, next_used_mask)) % MOD
                        
            memo[(last_index, used_mask)] = count
            return count
            
        return count_special_permutations(-1, 0)