class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        """
        We use a minimax + bitmask DP approach:
          1) Precompute the number of moves (shortest knight distance) between all
             relevant squares (the knight's initial position plus each pawn) using BFS.
          2) Let there be N pawns. We index them as 1..N in a "positions" list, while index 0 
             represents the knight's initial position. Thus we have N+1 total 'key' positions.
          3) We define a DP state dp(turn, i, mask) where
               - turn is 0 for Alice, 1 for Bob,
               - i is the current knight-position index (0 means the original knight pos,
                 1..N means that pawn's position),
               - mask is a bitmask of which pawns remain (e.g. if there are N pawns, then
                 a 1 in the p-th bit denotes that pawn p is still on the board).
             The value of dp is the optimal total number of knight moves from here to
             the end of the game when both sides play optimally for their objective
             (Alice maximizing, Bob minimizing).
          4) The transitions:
               - If mask == 0 (no pawns left), return 0.
               - If it's Alice's turn (turn=0), she chooses any available pawn p to maximize
                 dist[i][p+1] + dp(1, p+1, mask without p).
               - If it's Bob's turn (turn=1), he chooses any available pawn p to minimize
                 dist[i][p+1] + dp(0, p+1, mask without p).
          5) The answer is dp(0, 0, (1<<N)-1).
        """
        import sys
        sys.setrecursionlimit(10**7)
        from collections import deque

        # All the relevant positions: index 0 is knight's initial position,
        # indices 1..N are the pawns' positions.
        all_positions = [(kx, ky)] + positions
        N = len(positions)
        pos_count = N + 1  # total positions including knight start

        # Precompute distances dist[i][j] = min knight-moves from all_positions[i] to all_positions[j]
        dist = [[0]*pos_count for _ in range(pos_count)]

        # Knight's 8 possible moves
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                        (1, 2), (1, -2), (-1, 2), (-1, -2)]

        # BFS to get distances from a single start square to every other square
        def bfs(start_x, start_y):
            board_dist = [[-1]*50 for _ in range(50)]
            queue = deque()
            queue.append((start_x, start_y))
            board_dist[start_x][start_y] = 0
            while queue:
                x, y = queue.popleft()
                for dx, dy in knight_moves:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and board_dist[nx][ny] == -1:
                        board_dist[nx][ny] = board_dist[x][y] + 1
                        queue.append((nx, ny))
            return board_dist

        # Compute the pairwise distances using BFS for each relevant position
        for i in range(pos_count):
            sx, sy = all_positions[i]
            board_dist = bfs(sx, sy)
            for j in range(pos_count):
                tx, ty = all_positions[j]
                dist[i][j] = board_dist[tx][ty]

        # Prepare DP array (turn in [0,1], i in [0..N], mask in [0..(1<<N)-1])
        # We'll store them in a 1D list for speed: index = turn*(N+1)*(1<<N) + i*(1<<N) + mask
        SIZE = (N+1) * (1 << N) * 2
        dp = [-1]*SIZE

        # Helper to get dp index
        def idx(turn, i, mask):
            return turn*(N+1)*(1<<N) + i*(1<<N) + mask

        def solve_dp(i, mask, turn):
            # If no pawns left
            if mask == 0:
                return 0
            dp_index = idx(turn, i, mask)
            if dp[dp_index] != -1:
                return dp[dp_index]

            if turn == 0:
                # Alice's turn - maximize
                best = 0
                bitset = mask
                while bitset:
                    p = bitset & -bitset  # lowest set bit
                    pidx = (p).bit_length() - 1
                    # distance from current i to pawn pidx => dist[i][ pidx+1 ]
                    cand = dist[i][pidx+1] + solve_dp(pidx+1, mask ^ (1 << pidx), 1)
                    if cand > best:
                        best = cand
                    bitset ^= p
                dp[dp_index] = best
            else:
                # Bob's turn - minimize
                best = float('inf')
                bitset = mask
                while bitset:
                    p = bitset & -bitset  # lowest set bit
                    pidx = (p).bit_length() - 1
                    cand = dist[i][pidx+1] + solve_dp(pidx+1, mask ^ (1 << pidx), 0)
                    if cand < best:
                        best = cand
                    bitset ^= p
                dp[dp_index] = best

            return dp[dp_index]

        # All pawns initially present => mask = (1<<N) - 1
        # Start from the knight's position (index 0), with turn=0 (Alice).
        full_mask = (1 << N) - 1
        return solve_dp(0, full_mask, 0)