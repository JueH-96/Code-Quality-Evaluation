class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Build a list of candidates, where candidate^x is not greater than n
        candidates = []
        i = 1
        while i**x <= n:
            candidates.append(i**x)
            i += 1
        
        memo = {}
        
        def dfs(index, target):
            # when target reaches 0, we found a valid combination
            if target == 0:
                return 1
            # if target becomes negative or we have no more candidates, no valid combination
            if target < 0 or index == len(candidates):
                return 0
            
            # Check memoization
            if (index, target) in memo:
                return memo[(index, target)]
            
            # two choices: either use the current candidate or skip it
            # Use candidate at index:
            take = dfs(index + 1, target - candidates[index])
            # Skip candidate at index:
            skip = dfs(index + 1, target)
            memo[(index, target)] = (take + skip) % MOD
            return memo[(index, target)]
        
        return dfs(0, n)