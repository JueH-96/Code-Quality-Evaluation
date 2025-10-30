def find_recent_positions(N, A):
    B = [-1] * N
    last_seen = {}

    for i in range(N):
        if A[i] in last_seen:
            B[i] = last_seen[A[i]]
        last_seen[A[i]] = i + 1  # Store the position as 1-based index

    return B

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    result = find_recent_positions(N, A)
    print(" ".join(map(str, result)))