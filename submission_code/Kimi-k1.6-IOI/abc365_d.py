def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1] if len(input) > 1 else ''
    
    char_to_idx = {'R': 0, 'P': 1, 'S': 2}
    aoki = [char_to_idx[c] for c in S]
    
    # moves[aoki_idx] gives list of (takahashi_move, win_value)
    moves = [
        [(1, 1), (0, 0)],  # Aoki plays R
        [(2, 1), (1, 0)],  # Aoki plays P
        [(0, 1), (2, 0)]   # Aoki plays S
    ]
    
    if N == 0:
        print(0)
        return
    
    prev_dp = [float('-inf')] * 3
    first_move = aoki[0]
    for m, w in moves[first_move]:
        prev_dp[m] = w
    
    for i in range(1, N):
        curr_aoki = aoki[i]
        curr_dp = [float('-inf')] * 3
        for m, w in moves[curr_aoki]:
            m1 = (m + 1) % 3
            m2 = (m + 2) % 3
            max_prev = max(prev_dp[m1], prev_dp[m2])
            curr_dp[m] = max_prev + w
        prev_dp = curr_dp
    
    print(max(prev_dp))

if __name__ == "__main__":
    main()