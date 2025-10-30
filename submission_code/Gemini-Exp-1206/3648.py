class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        dp = {}

        def solve(r1, c1, r2, c2, r3, c3):
            if (r1, c1, r2, c2, r3, c3) in dp:
                return dp[(r1, c1, r2, c2, r3, c3)]

            if r1 == n - 1 and c1 == n - 1 and r2 == n - 1 and c2 == n - 1 and r3 == n - 1 and c3 == n - 1:
                return fruits[r1][c1]

            ans = 0
            
            collected = fruits[r1][c1]
            if (r1, c1) != (r2, c2):
                collected += fruits[r2][c2]
            if (r1, c1) != (r3, c3) and (r2, c2) != (r3, c3):
                collected += fruits[r3][c3]
                
            
            moves1 = []
            if r1 + 1 < n and c1 + 1 < n:
                moves1.append((r1 + 1, c1 + 1))
            if r1 + 1 < n:
                moves1.append((r1 + 1, c1))
            if c1 + 1 < n:
                moves1.append((r1, c1 + 1))

            moves2 = []
            if r2 + 1 < n and c2 - 1 >= 0:
                moves2.append((r2 + 1, c2 - 1))
            if r2 + 1 < n:
                moves2.append((r2 + 1, c2))
            if r2 + 1 < n and c2 + 1 < n:
                moves2.append((r2 + 1, c2 + 1))
                
            moves3 = []
            if r3 - 1 >= 0 and c3 + 1 < n:
                moves3.append((r3 - 1, c3 + 1))
            if c3 + 1 < n:
                moves3.append((r3, c3 + 1))
            if r3 + 1 < n and c3 + 1 < n:
                moves3.append((r3 + 1, c3 + 1))

            for nr1, nc1 in moves1:
                for nr2, nc2 in moves2:
                    for nr3, nc3 in moves3:
                        ans = max(ans, solve(nr1, nc1, nr2, nc2, nr3, nc3))

            dp[(r1, c1, r2, c2, r3, c3)] = ans + collected
            return dp[(r1, c1, r2, c2, r3, c3)]

        return solve(0, 0, 0, n - 1, n - 1, 0)