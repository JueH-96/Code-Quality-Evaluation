def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = input[idx]
    idx += 1
    C = list(map(int, input[idx:idx+N]))
    
    if N == 1:
        print(0)
        return
    
    left_0 = [0] * N
    left_1 = [0] * N
    
    # Initialize left arrays
    left_0[0] = 0 if S[0] == '0' else C[0]
    left_1[0] = 0 if S[0] == '1' else C[0]
    
    for j in range(1, N):
        left_0[j] = left_1[j-1] + (0 if S[j] == '0' else C[j])
        left_1[j] = left_0[j-1] + (0 if S[j] == '1' else C[j])
    
    right_0 = [0] * N
    right_1 = [0] * N
    
    # Initialize right arrays
    right_0[N-1] = 0 if S[N-1] == '0' else C[N-1]
    right_1[N-1] = 0 if S[N-1] == '1' else C[N-1]
    
    for j in range(N-2, -1, -1):
        right_0[j] = (0 if S[j] == '0' else C[j]) + right_1[j+1]
        right_1[j] = (0 if S[j] == '1' else C[j]) + right_0[j+1]
    
    min_total = float('inf')
    for i in range(N-1):
        cost0 = left_0[i] + right_0[i+1]
        cost1 = left_1[i] + right_1[i+1]
        current_min = min(cost0, cost1)
        if current_min < min_total:
            min_total = current_min
    
    print(min_total)

if __name__ == '__main__':
    main()