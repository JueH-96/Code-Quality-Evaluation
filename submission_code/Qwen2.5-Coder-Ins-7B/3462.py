class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        n = len(s)
        dp = [0] * (n + 1)
        count = 0
        
        for i in range(n - 1, -1, -1):
            if s[i] in vowels:
                count += 1
            dp[i] = count % 2
        
        alice_turn = True
        for i in range(n):
            if dp[i] % 2 == 1:
                if alice_turn:
                    return True
                alice_turn = False
            else:
                if not alice_turn:
                    return True
                alice_turn = True
        
        return False