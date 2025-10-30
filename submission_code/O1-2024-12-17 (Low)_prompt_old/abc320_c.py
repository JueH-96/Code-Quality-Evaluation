def solve():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    M = int(input_data[0])
    S1 = input_data[1]
    S2 = input_data[2]
    S3 = input_data[3]

    # Precompute, for each digit d=0..9, which reel‐positions (mod M) yield that digit
    # pos[i][d] will be True if S_i has digit d at index i (0-based).
    # We'll store each reel as an array of length M of booleans:
    #   pos1[d][r] = True if S1[r] == str(d), else False, etc.
    
    # But for easier membership checks, we can store sets of valid residues for each digit.
    # For reel 1 (S1), digit d => set of r s.t. S1[r] == str(d).
    # Similarly for S2, S3.
    
    reels = [S1, S2, S3]
    pos = []
    for reel in reels:
        # pos_for_reel[d] = set of r where reel[r] == str(d)
        pos_for_reel = [set() for _ in range(10)]
        for r, ch in enumerate(reel):
            digit = int(ch)
            pos_for_reel[digit].add(r)
        pos.append(pos_for_reel)

    # We will attempt each digit d in [0..9].
    # If all three reels have at least one position that shows d, we do a BFS over time up to 3*M
    # to see if we can press the three reels (in distinct times) so that each reel lands on d.
    #
    # The BFS state is (t, bits) where:
    #   - t is the current time (0 <= t <= 3*M),
    #   - bits is a 3-bit mask indicating which reels are already pressed (p1, p2, p3).
    #   p1 is bit 2, p2 is bit 1, p3 is bit 0.
    #
    # Transitions from state (t, bits):
    #   1) do nothing => (t+1, bits)   (if t < 3*M)
    #   2) if reel1 not pressed and (t mod M) in pos[0][d], we can press it => (t+1, bits|4)
    #      similarly for reel2 => bits|2, reel3 => bits|1
    #
    # As soon as we get bits == 7 (all pressed) in BFS at time t, we return t as the finishing time.
    #
    # We'll do this BFS for each d. The answer is the minimum t found among all d, or -1 if none.

    def bfs_for_digit(d):
        # If any reel has no position for digit d, impossible immediately
        if not pos[0][d] or not pos[1][d] or not pos[2][d]:
            return None

        max_time = 3 * M  # We'll search up to this time
        visited = [[False]*8 for _ in range(max_time+1)]
        # visited[t][bits] = True means we've visited time t with pressed-bits = bits
        # bits in [0..7], p1=bit2, p2=bit1, p3=bit0

        from collections import deque
        queue = deque()
        # Start at time 0 with none pressed
        visited[0][0] = True
        queue.append((0, 0))  # (time, bits)

        while queue:
            t, bits = queue.popleft()
            p1 = (bits >> 2) & 1
            p2 = (bits >> 1) & 1
            p3 = bits & 1

            # If all pressed, we're done
            if p1 == 1 and p2 == 1 and p3 == 1:
                return t  # the earliest time we reach all pressed

            if t == max_time:
                # We do not go beyond max_time
                continue

            next_t = t + 1

            # 1) Do nothing
            if not visited[next_t][bits]:
                visited[next_t][bits] = True
                queue.append((next_t, bits))

            # 2) Press each reel if not pressed yet and if digit d is valid at t mod M
            r = t % M
            # reel1
            if p1 == 0 and r in pos[0][d]:
                new_bits = bits | 4  # set bit2
                if not visited[next_t][new_bits]:
                    visited[next_t][new_bits] = True
                    queue.append((next_t, new_bits))
            # reel2
            if p2 == 0 and r in pos[1][d]:
                new_bits = bits | 2  # set bit1
                if not visited[next_t][new_bits]:
                    visited[next_t][new_bits] = True
                    queue.append((next_t, new_bits))
            # reel3
            if p3 == 0 and r in pos[2][d]:
                new_bits = bits | 1  # set bit0
                if not visited[next_t][new_bits]:
                    visited[next_t][new_bits] = True
                    queue.append((next_t, new_bits))

        # If we exhaust BFS without pressing all reels, return None
        return None

    ans = None
    for d in range(10):
        res = bfs_for_digit(d)
        if res is not None:
            if ans is None or res < ans:
                ans = res

    if ans is None:
        print(-1)
    else:
        print(ans)