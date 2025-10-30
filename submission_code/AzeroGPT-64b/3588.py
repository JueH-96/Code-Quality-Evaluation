MOD = 1000000007

def score(a, b):
    if a == b:
        return 0, 0
    elif (a, b) in [('F', 'W'), ('W', 'E'), ('E', 'F')]:
        return 0, 1
    else:
        return 1, 0

tonum = {'F': 0, 'W': 1, 'E': 2}
lookup = {(i, j): 0 for i in range(3) for j in range(3)}

lookup[0, 1] = lookup[1, 2] = lookup[2, 0] = 1

class Solution:
    def countWinningSequences(self, s: str) -> int:
        a = tonum[s[0]]
        ans = [0, 0]
        for b in range(3):
            if b != a:
                ans[0] += lookup[(a, b)]
                ans[1] += MAX - lookup[(a, b)] + 1
        lookup[(a, a)] = 1
        for ch in s[1:]:
            a = tonum[ch]
            c, d = ans
            ans = [0, 0]
            for b in range(3):
                if b != a:
                    ans[0] += (c + lookup[(a, b)]) * (MAX - lookup[(b, a)] + 1) % MOD
                    ans[1] += (d + MAX - lookup[(a, b)] + 1) * lookup[(b, a)] % MOD
            lookup[(a, a)] = 1
            for b in range(3):
                if b != a:
                    lookup[(a, b)] = (lookup[(a, b)] + c) % MOD

        return ((ans[0] - ans[1]) % MOD + MOD) % MOD