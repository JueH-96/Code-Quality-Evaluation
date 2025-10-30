def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = input[idx]
    idx += 1
    C = list(map(int, input[idx:idx+N]))
    
    # Convert S to integers for easier handling
    S_int = [int(c) for c in S]
    
    # Initialize left array
    left = [[float('inf')] * 2 for _ in range(N)]
    # Base case: i=0
    if S_int[0] == 0:
        left[0][0] = 0
    else:
        left[0][1] = 0
    if S_int[0] == 1:
        left[0][1] = 0
    else:
        left[0][0] = C[0]
    
    for i in range(1, N):
        for b in [0, 1]:
            # We can come from the opposite bit
            prev_b = 1 - b
            cost = C[i] if (S_int[i] != b) else 0
            left[i][b] = min(left[i-1][prev_b] + cost)
    
    # Initialize right array
    right = [[float('inf')] * 2 for _ in range(N)]
    for b in [0, 1]:
        if S_int[-1] == b:
            right[-1][b] = 0
        else:
            right[-1][b] = C[-1]
    
    for i in range(N-2, -1, -1):
        for b in [0, 1]:
            # The next bit must be different
            next_b = 1 - b
            cost_flip = C[i+1] if (S_int[i+1] != next_b) else 0
            total_cost = cost_flip + right[i+1][next_b]
            right[i][b] = total_cost
    
    min_total = float('inf')
    for i in range(N-1):
        for b in [0, 1]:
            # Calculate the cost to set T[i] = b and T[i+1] = b
            # Left part: up to i
            if left[i][b] == float('inf'):
                continue
            # Cost to set T[i+1] = b
            if S_int[i+1] == b:
                cost_i_plus_1 = 0
            else:
                cost_i_plus_1 = C[i+1]
            
            # Right part: starting from i+2
            if i + 2 < N:
                # Need T[i+2] != b
                min_right = min(right[i+2][1 - b])
            else:
                min_right = 0
            
            total = left[i][b] + cost_i_plus_1 + min_right
            if total < min_total:
                min_total = total
    
    print(min_total if min_total != float('inf') else 0)

if __name__ == '__main__':
    main()