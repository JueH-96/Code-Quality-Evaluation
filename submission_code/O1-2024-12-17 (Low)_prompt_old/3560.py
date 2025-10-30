class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        from collections import deque

        # All candidate squares: 
        # index 0 is the initial knight position,
        # indices 1..n are the n pawn positions.
        squares = [(kx, ky)] + positions  
        n = len(squares)

        # Precompute knight distances between every pair of these squares (BFS)
        # Since n <= 16, we can store distances in a matrix dist[i][j].
        dist = [[0]*n for _ in range(n)]
        board_size = 50
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)]

        def bfs(start_index):
            # Returns an array of minimal distances from squares[start_index]
            # to squares[any_index].
            start = squares[start_index]
            queue = deque([(start[0], start[1], 0)])
            visited = [[False]*board_size for _ in range(board_size)]
            visited[start[0]][start[1]] = True
            res = [0]*n
            # once we reach squares[i], we record the distance
            found_count = 1  # we've "found" the start itself
            while queue:
                x, y, steps = queue.popleft()
                # If (x, y) is one of the squares of interest, record distance
                # and possibly terminate if we've found all.
                for i in range(n):
                    if (x, y) == squares[i]:
                        res[i] = steps
                # BFS expansion
                for dx, dy in knight_moves:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < board_size and 0 <= ny < board_size:
                        if not visited[nx][ny]:
                            visited[nx][ny] = True
                            queue.append((nx, ny, steps+1))
            return res

        for i in range(n):
            dist_i = bfs(i)
            for j in range(n):
                dist[i][j] = dist_i[j]

        # We have at most 15 pawns -> bitmask of length up to 15
        # The DP state: dp(currentSquare, bitmask, turn)
        #   currentSquare: 0..n-1  (knight currently at squares[currentSquare])
        #   bitmask: which pawns are still available
        #            We will use bits 0..(n-2) to represent pawn indices 1..(n-1).
        #   turn: 0 for Alice's turn (maximizer), 1 for Bob's turn (minimizer)
        # The value of dp is the maximum/minimum total number of moves from this state
        # until all pawns are taken.

        from functools import lru_cache

        # Pre-shift: We store pawns in bits 0..(len(positions)-1).
        # squares[1 + i] corresponds to bit i in the mask.
        # If the i-th bit is set, it means squares[i+1] is still on the board.

        all_pawns_mask = (1 << len(positions)) - 1

        @lru_cache(None)
        def dfs(current, mask, turn):
            # If no pawns left, no more moves can be made
            if mask == 0:
                return 0

            if turn == 0:
                # Alice's turn: she will pick a pawn to maximize the total moves
                best = 0
                # Try capturing each available pawn
                b = mask
                while b:
                    # p is the index of a pawn that is currently on the board
                    p = (b & -b).bit_length() - 1  # get rightmost set bit
                    b ^= (1 << p)  # remove that bit from consideration
                    # distance from current to the chosen pawn
                    moves = dist[current][p+1]   # squares indices offset
                    # new state
                    next_mask = mask & ~(1 << p)
                    # Bob's turn next
                    val = moves + dfs(p+1, next_mask, 1)
                    best = max(best, val)
                return best
            else:
                # Bob's turn: he will pick a pawn to minimize the total moves
                worst = float('inf')
                b = mask
                while b:
                    p = (b & -b).bit_length() - 1
                    b ^= (1 << p)
                    moves = dist[current][p+1]
                    next_mask = mask & ~(1 << p)
                    # Alice's turn next
                    val = moves + dfs(p+1, next_mask, 0)
                    worst = min(worst, val)
                return worst

        # Start from squares[0], all pawns are present, turn = 0 (Alice)
        return dfs(0, all_pawns_mask, 0)