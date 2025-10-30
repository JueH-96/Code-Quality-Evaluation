class Solution:
    def punishmentNumber(self, n: int) -> int:
        def check(num, target):
            s = str(num)
            
            def backtrack(index, current_sum):
                if index == len(s):
                    return current_sum == target
                
                for i in range(index, len(s)):
                    val = int(s[index:i+1])
                    if backtrack(i+1, current_sum + val):
                        return True
                return False
            
            return backtrack(0, 0)

        ans = 0
        for i in range(1, n + 1):
            sq = i * i
            if check(sq, i):
                ans += sq
        return ans