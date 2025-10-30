class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        dp = {}

        def solve(r1, c1, r2, c2, r3, c3, moves):
            if moves == n:
                return 0

            if (r1, c1, r2, c2, r3, c3, moves) in dp:
                return dp[(r1, c1, r2, c2, r3, c3, moves)]

            ans = 0
            
            # Child 1 moves
            moves1 = []
            if r1 + 1 < n and c1 + 1 < n:
                moves1.append((r1 + 1, c1 + 1))
            if r1 + 1 < n:
                moves1.append((r1 + 1, c1))
            if c1 + 1 < n:
                moves1.append((r1, c1 + 1))

            # Child 2 moves
            moves2 = []
            if r2 + 1 < n and c2 - 1 >= 0:
                moves2.append((r2 + 1, c2 - 1))
            if r2 + 1 < n:
                moves2.append((r2 + 1, c2))
            if r2 + 1 < n and c2 + 1 < n:
                moves2.append((r2 + 1, c2 + 1))
            
            # Child 3 moves
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
                        curr_fruits = 0
                        visited = set()
                        if (nr1, nc1) not in visited:
                            curr_fruits += fruits[nr1][nc1]
                            visited.add((nr1, nc1))
                        if (nr2, nc2) not in visited:
                            curr_fruits += fruits[nr2][nc2]
                            visited.add((nr2, nc2))
                        if (nr3, nc3) not in visited:
                            curr_fruits += fruits[nr3][nc3]
                            visited.add((nr3, nc3))
                        
                        ans = max(ans, curr_fruits + solve(nr1, nc1, nr2, nc2, nr3, nc3, moves + 1))
            
            dp[(r1, c1, r2, c2, r3, c3, moves)] = ans
            return ans

        return solve(0, 0, 0, n - 1, n - 1, 0, 0)