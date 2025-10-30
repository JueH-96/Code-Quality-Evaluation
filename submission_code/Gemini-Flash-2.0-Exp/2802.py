class Solution:
    def punishmentNumber(self, n: int) -> int:
        def check(num: int) -> bool:
            s = str(num * num)
            
            def backtrack(index: int, current_sum: int) -> bool:
                if index == len(s):
                    return current_sum == num
                
                for i in range(index, len(s)):
                    val = int(s[index:i+1])
                    if backtrack(i + 1, current_sum + val):
                        return True
                
                return False
            
            return backtrack(0, 0)
        
        ans = 0
        for i in range(1, n + 1):
            if check(i):
                ans += i * i
        
        return ans