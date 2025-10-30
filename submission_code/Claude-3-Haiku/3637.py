class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num
        MOD = 10**9 + 7
        n = len(num)
        
        def backtrack(perm, used):
            if len(perm) == n:
                if sum(int(perm[i]) for i in range(0, n, 2)) == sum(int(perm[i]) for i in range(1, n, 2)):
                    return 1
                return 0
            
            count = 0
            for i in range(n):
                if not used[i]:
                    used[i] = True
                    perm.append(num[i])
                    count += backtrack(perm, used)
                    used[i] = False
                    perm.pop()
            return count
        
        used = [False] * len(num)
        return sum(backtrack([num[i]], used) for i in range(len(num))) % MOD