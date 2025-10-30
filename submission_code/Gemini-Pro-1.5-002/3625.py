class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n <= 9:
            return False
        
        dp = [False] * (n + 1)

        for stones in range(1, n + 1):
            if stones <= 9:
                dp[stones] = False
            elif stones == 10:
                dp[stones] = True
            else:
                moves = []
                for prev_move in range(1, 11):
                    if stones - prev_move >= 0:
                        moves.append(stones - prev_move)
                
                can_win = False
                for move in moves:
                    if move >= 1 and not dp[move]:
                        can_win = True
                        break
                dp[stones] = can_win
        
        
        if n >= 10:
            moves = []
            for prev_move in range(1, 11):
                if n - prev_move >= 0:
                    moves.append(n - prev_move)
            can_win = False
            for move in moves:
                if not dp[move]:
                    can_win = True
                    break
            return can_win
        else:
            return False