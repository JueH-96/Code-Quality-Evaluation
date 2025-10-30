class Solution:
    def doesAliceWin(self, s: str) -> bool:
        cnt = [0,0]
        pre = 0
        cnt[pre] +=1  # pre[0]=0
        vowels = {'a','e','i','o','u'}
        for ch in s:
            if ch in vowels:
                pre ^=1  # Flip parity
            cnt[pre] +=1
        odd_substrings = cnt[0]*cnt[1]
        if odd_substrings %2 ==1:
            return True  # Alice wins
        else:
            return False  # Alice loses