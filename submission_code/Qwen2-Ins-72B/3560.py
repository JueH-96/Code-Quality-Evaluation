class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        def knight_moves(x, y):
            for dx, dy in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                yield x + dx, y + dy

        @cache
        def dp(turn, mask):
            if turn == len(positions):
                return 0

            x, y = positions[turn]
            ans = 0
            for nx, ny in knight_moves(x, y):
                if 0 <= nx < 50 and 0 <= ny < 50 and mask & (1 << (nx * 50 + ny)) == 0:
                    ans = max(ans, 1 + dp(turn + 1, mask | (1 << (nx * 50 + ny))))
            if turn % 2 == 0:
                ans = max(ans, dp(turn + 1, mask))
            else:
                ans = min(ans, dp(turn + 1, mask))
            return ans

        return dp(0, 1 << (kx * 50 + ky))