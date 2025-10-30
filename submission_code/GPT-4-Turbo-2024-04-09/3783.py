from itertools import permutations

class Solution:
    def permute(self, n: int, k: int) -> list:
        def is_alternating(perm):
            return all((perm[i] % 2) != (perm[i+1] % 2) for i in range(len(perm) - 1))
        
        nums = list(range(1, n + 1))
        all_perms = permutations(nums)
        valid_perms = []
        
        for perm in all_perms:
            if is_alternating(perm):
                valid_perms.append(perm)
                if len(valid_perms) == k:
                    return list(perm)
        
        return [] if len(valid_perms) < k else list(valid_perms[k-1])

# Example usage:
sol = Solution()
print(sol.permute(4, 6))  # Output: [3, 4, 1, 2]
print(sol.permute(3, 2))  # Output: [3, 2, 1]
print(sol.permute(2, 3))  # Output: []