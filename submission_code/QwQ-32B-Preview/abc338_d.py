def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:2+M]))
    
    # Function to compute distance between two islands
    def d(a, b):
        return min((b - a) % N, (a - b) % N)
    
    # Initialize C as an array of size N+2 for prefix sums
    C = [0] * (N + 2)
    
    total = 0
    for i in range(M - 1):
        a = X[i] - 1  # 0-indexed
        b = X[i + 1] - 1  # 0-indexed
        dist = d(a, b)
        D_i = min(dist, N - dist)
        total += D_i
        
        # Determine shorter path
        if dist < N - dist:
            # Clockwise
            if a < b:
                # Add to C[a+1] and subtract from C[b+1]
                val = N - 2 * D_i
                C[a + 1] += val
                C[b + 1] -= val
            else:
                # Add to C[a+1] and C[1], subtract from C[N+1] and C[b+1]
                val = N - 2 * D_i
                C[a + 1] += val
                C[N + 1] -= val
                C[1] += val
                C[b + 1] -= val
        else:
            # Counterclockwise
            if b < a:
                # Add to C[b+1], subtract from C[a+1]
                val = N - 2 * D_i
                C[b + 1] += val
                C[a + 1] -= val
            else:
                # Add to C[b+1] and C[1], subtract from C[N+1] and C[a+1]
                val = N - 2 * D_i
                C[b + 1] += val
                C[N + 1] -= val
                C[1] += val
                C[a + 1] -= val
    
    # Compute prefix sums
    for i in range(1, N + 1):
        C[i] += C[i - 1]
    for i in range(1, N + 2):
        C[i] += C[i - 1]
    
    # Compute sum_S(B) for each B and find the minimum
    min_sum = float('inf')
    for B in range(1, N + 1):
        sum_S_B = total + C[B]
        if sum_S_B < min_sum:
            min_sum = sum_S_B
    
    print(min_sum)

if __name__ == '__main__':
    main()