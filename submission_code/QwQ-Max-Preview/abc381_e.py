import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    S = data[idx]
    idx += 1
    
    # Precompute prefix_ones: prefix_ones[i] is the number of 1s in S[0..i-1]
    prefix_ones = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_ones[i] = prefix_ones[i-1] + (1 if S[i-1] == '1' else 0)
    
    # Precompute suffix_twos: suffix_twos[i] is the number of 2s in S[i..N-1]
    suffix_twos = [0] * (N + 1)
    for i in range(N-1, -1, -1):
        suffix_twos[i] = suffix_twos[i+1] + (1 if S[i] == '2' else 0)
    
    # Collect all '/' positions with their a_i and b_i
    slash_positions = []
    for pos in range(N):
        if S[pos] == '/':
            a_i = prefix_ones[pos]
            b_i = suffix_twos[pos + 1]
            slash_positions.append((pos, a_i, b_i))
    
    # Extract pos from slash_positions for binary search
    slash_pos_list = [p[0] for p in slash_positions]
    
    for _ in range(Q):
        L = int(data[idx]) - 1  # convert to 0-based
        idx += 1
        R = int(data[idx]) - 1
        idx += 1
        
        x = prefix_ones[L]
        y = suffix_twos[R + 1]
        
        # Find all slash positions in [L, R]
        if not slash_pos_list:
            print(0)
            continue
        
        left = bisect.bisect_left(slash_pos_list, L)
        right = bisect.bisect_right(slash_pos_list, R) - 1
        
        if left > right:
            print(0)
            continue
        
        max_k = -1
        # Iterate over the relevant slash positions
        for i in range(left, right + 1):
            pos, a_i, b_i = slash_positions[i]
            available_1 = a_i - x
            available_2 = b_i - y
            if available_1 < 0 or available_2 < 0:
                continue
            current_k = min(available_1, available_2)
            if current_k > max_k:
                max_k = current_k
        
        if max_k == -1:
            print(0)
        else:
            print(2 * max_k + 1)

if __name__ == "__main__":
    main()