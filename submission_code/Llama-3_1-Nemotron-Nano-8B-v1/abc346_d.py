def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = input[idx]
    idx += 1
    C = list(map(int, input[idx:idx+N]))
    
    # Precompute left and right arrays
    left = [[0] * N for _ in range(2)]
    for j in range(N):
        for s in 0, 1:
            if j == 0:
                cost = 0 if S[j] == str(s) else C[j]
            else:
                prev_s = 1 - s
                cost = left[prev_s][j-1] + (0 if S[j] == str(s) else C[j])
            left[s][j] = cost
    
    right = [[0] * N for _ in range(2)]
    for j in range(N-1, -1, -1):
        for s in 0, 1:
            if j == N-1:
                cost = 0 if S[j] == str(s) else C[j]
            else:
                next_s = 1 - s
                cost = right[next_s][j+1] + (0 if S[j] == str(s) else C[j])
            right[s][j] = cost
    
    ans = float('inf')
    for i in range(N-1):
        for a in 0, 1:
            # Calculate flip cost for i and i+1 to a
            cost_flip = (0 if S[i] == str(a) else C[i]) + (0 if S[i+1] == str(a) else C[i+1])
            
            # Calculate left cost
            left_possible = []
            if i == 0:
                left_possible.append(0)
            else:
                for s in 0, 1:
                    last_bit = s if ((i-1) % 2 == 0) else (1 - s)
                    if last_bit != a:
                        left_possible.append(left[s][i-1])
            if not left_possible:
                continue
            left_cost = min(left_possible)
            
            # Calculate right cost
            right_start = 1 - a
            right_part_start = i + 2
            if right_part_start >= N:
                right_cost = 0
            else:
                right_cost = right[right_start][right_part_start]
            
            total = cost_flip + left_cost + right_cost
            if total < ans:
                ans = total
    print(ans)

if __name__ == "__main__":
    main()