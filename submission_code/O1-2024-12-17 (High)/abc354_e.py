def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = [0]*N
    B = [0]*N

    idx = 1
    for i in range(N):
        A[i] = int(data[idx]); B[i] = int(data[idx+1])
        idx += 2

    # Precompute adjacency via bitmask: comp[i] holds a bitmask of cards j
    # that can be paired with i (sharing either front or back value).
    comp = [0]*N
    for i in range(N):
        for j in range(i+1, N):
            if A[i] == A[j] or B[i] == B[j]:
                comp[i] |= (1 << j)
                comp[j] |= (1 << i)

    # dp[state] = -1 means not computed, 0 means losing position,
    # 1 means winning position.
    dp = [-1] * (1 << N)

    def whoWins(state):
        # If no cards remain, current player cannot move -> losing.
        if state == 0:
            return 0
        if dp[state] != -1:
            return dp[state]
        
        # Pick the lowest set bit i in "state" (first card to try pairing).
        i_bit = state & -state
        i = i_bit.bit_length() - 1
        
        # Possible partners for card i must be in state and in comp[i].
        jMask = comp[i] & state & ~(1 << i)
        
        # Try each possible partner j. If removing i,j leads opponent to lose,
        # then current player has a winning move.
        x = jMask
        while x:
            j_bit = x & -x
            j = j_bit.bit_length() - 1

            newstate = state ^ i_bit ^ j_bit
            if whoWins(newstate) == 0:
                dp[state] = 1
                return 1
            
            x ^= j_bit
        
        dp[state] = 0
        return 0

    initial_state = (1 << N) - 1
    if whoWins(initial_state) == 1:
        print("Takahashi")
    else:
        print("Aoki")

# Do not forget to call main()
if __name__ == "__main__":
    main()