import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    
    N = int(data[idx])
    idx += 1
    
    X = list(map(int, data[idx:idx+N]))
    idx += N
    
    Q = int(data[idx])
    idx += 1
    
    last_query = [-1] * N  # -1 means no query
    for _ in range(Q):
        T = int(data[idx]) - 1  # 0-based
        G = int(data[idx+1])
        idx += 2
        last_query[T] = G
    
    # Initialize final_Y
    final_Y = [0] * N
    for i in range(N):
        if last_query[i] != -1:
            final_Y[i] = last_query[i]
        else:
            final_Y[i] = X[i]
    
    # Process left to right
    for i in range(1, N):
        min_allowed = final_Y[i-1] + 1
        if final_Y[i] < min_allowed:
            final_Y[i] = min_allowed
    
    # Process right to left
    for i in range(N-2, -1, -1):
        max_allowed = final_Y[i+1] - 1
        if final_Y[i] > max_allowed:
            final_Y[i] = max_allowed
    
    # Calculate total movements
    total = 0
    for i in range(N):
        total += abs(final_Y[i] - X[i])
    
    print(total)

if __name__ == "__main__":
    main()