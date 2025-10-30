class Solution:
    def punishmentNumber(self, n: int) -> int:
        # Helper function to check if s can be partitioned to sum up to target.
        def can_partition(s: str, target: int) -> bool:
            m = len(s)
            
            # cache for memoization: (pos, remaining) -> bool
            memo = {}
            
            def dfs(pos: int, remaining: int) -> bool:
                # Base case: if we've reached the end of string
                if pos == m:
                    return remaining == 0
                if (pos, remaining) in memo:
                    return memo[(pos, remaining)]
                
                # Try all possible splits from pos to end
                curr_num = 0
                for j in range(pos, m):
                    # Build the number from s[pos:j+1]
                    curr_num = curr_num * 10 + int(s[j])
                    if curr_num > remaining:
                        # Since further digits will only increase curr_num, break early.
                        break
                    # If the recursive call finds a valid partition, return True.
                    if dfs(j + 1, remaining - curr_num):
                        memo[(pos, remaining)] = True
                        return True
                memo[(pos, remaining)] = False
                return False
            
            return dfs(0, target)
        
        result = 0
        for i in range(1, n + 1):
            ss = str(i * i)
            if can_partition(ss, i):
                result += i * i
        return result