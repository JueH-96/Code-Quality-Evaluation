def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1

    # Read P points
    P = []
    for i in range(1, N+1):
        a = int(input[idx])
        b = int(input[idx+1])
        idx +=2
        P.append( (a, b, i) )

    # Read Q points
    Q = []
    for i in range(1, N+1):
        c = int(input[idx])
        d = int(input[idx+1])
        idx +=2
        Q.append( (c, d, i) )

    # Sort P and Q by their coordinates
    sorted_P = sorted(P, key=lambda x: (x[0], x[1]))
    sorted_Q = sorted(Q, key=lambda x: (x[0], x[1]))

    # Create the result array
    R = [0] * (N + 1)  # 1-based indexing
    for i in range(N):
        p = sorted_P[i]
        q = sorted_Q[i]
        original_p = p[2]
        original_q = q[2]
        R[original_p] = original_q

    # Output the result
    print(' '.join(map(str, R[1:N+1])))

if __name__ == "__main__":
    main()