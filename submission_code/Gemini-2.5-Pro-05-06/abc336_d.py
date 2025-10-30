import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # l[i]: max height k_L such that A[i] can be the k_L-th term of 1, ..., k_L
    l = [0] * N
    # Base case: l[0] = min(A[0], 0+1) = 1, since A[i] >= 1.
    l[0] = 1 
    for i in range(1, N):
        l[i] = min(A[i], l[i-1] + 1)

    # r[i]: max height k_R such that A[i] can be the k_R-th term of k_R, ..., 1
    r = [0] * N
    # Base case: r[N-1] = min(A[N-1], 0+1) = 1, since A[i] >= 1.
    r[N-1] = 1
    for i in range(N-2, -1, -1): # Iterate from N-2 down to 0
        r[i] = min(A[i], r[i+1] + 1)

    max_k = 0
    for i in range(N):
        # Max k for A[i] as peak is min(l[i], r[i])
        current_k_for_peak_i = min(l[i], r[i])
        if current_k_for_peak_i > max_k:
            max_k = current_k_for_peak_i
            
    print(max_k)

if __name__ == '__main__':
    main()