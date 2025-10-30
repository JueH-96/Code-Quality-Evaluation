def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = input[idx]
    idx += 1
    C = list(map(int, input[idx:idx+N]))
    
    # Compute prefix_0 and prefix_1
    prefix_0 = [0] * N
    prefix_1 = [0] * N
    prefix_0[0] = 0 if S[0] == '0' else C[0]
    prefix_1[0] = 0 if S[0] == '1' else C[0]
    for j in range(1, N):
        expected = '0' if j % 2 == 0 else '1'
        prefix_0[j] = prefix_0[j-1] + (C[j] if S[j] != expected else 0)
        expected = '1' if j % 2 == 0 else '0'
        prefix_1[j] = prefix_1[j-1] + (C[j] if S[j] != expected else 0)
    
    # Compute suffix_0 and suffix_1
    suffix_0 = [0] * N
    suffix_1 = [0] * N
    for j in range(N-1, -1, -1):
        # suffix_0[j] = cost to start with 0 at j
        cost_0 = 0 if S[j] == '0' else C[j]
        if j == N-1:
            suffix_0[j] = cost_0
        else:
            suffix_0[j] = cost_0 + suffix_1[j+1]
        # suffix_1[j] = cost to start with 1 at j
        cost_1 = 0 if S[j] == '1' else C[j]
        if j == N-1:
            suffix_1[j] = cost_1
        else:
            suffix_1[j] = cost_1 + suffix_0[j+1]
    
    min_cost = float('inf')
    for i in range(N-1):
        # Case 00
        if i % 2 == 0:
            p_cost = prefix_0[i]
        else:
            p_cost = prefix_1[i]
        pair_cost = C[i+1] if S[i+1] != '0' else 0
        s_cost = suffix_1[i+2] if i+2 < N else 0
        total = p_cost + pair_cost + s_cost
        min_total = total
        
        # Case 11
        if i % 2 == 0:
            p_cost = prefix_1[i]
        else:
            p_cost = prefix_0[i]
        pair_cost = C[i+1] if S[i+1] != '1' else 0
        s_cost = suffix_0[i+2] if i+2 < N else 0
        total = p_cost + pair_cost + s_cost
        min_total = min(min_total, total)
        
        if min_total < min_cost:
            min_cost = min_total
    
    print(min_cost)

if __name__ == '__main__':
    main()