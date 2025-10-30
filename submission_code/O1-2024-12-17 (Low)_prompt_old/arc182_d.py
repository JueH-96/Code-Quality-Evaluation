def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))

    # If A is already B, no operations needed
    if A == B:
        print(0)
        return

    # Special case when M = 2
    # For M=2, the only good sequences are strictly alternating 0,1,0,1,... or 1,0,1,0,...
    # Changing any single element flips it from 0→1 or 1→0, which would immediately
    # match its neighbor if it was different before. Hence no valid single-step transformation
    # exists unless A == B from the start (already handled). So if A != B, it's impossible.
    if M == 2:
        print(-1)
        return

    # For M > 2, it can be shown that it is always possible to perform the transformation
    # without violating adjacency if B is also a good sequence. The minimum cost is simply
    # the sum of the individual circular distances between A_i and B_i.
    total_operations = 0
    for i in range(N):
        diff = abs(A[i] - B[i])
        # Since we're working modulo M, pick the minimal distance around the circle
        total_operations += min(diff, M - diff)
    
    print(total_operations)

def main():
    solve()

if __name__ == "__main__":
    main()