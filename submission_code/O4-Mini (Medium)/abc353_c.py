import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    M = 10**8
    A.sort()
    
    # Compute sum of all A_i
    total_sum = sum(A)
    
    # Count T = number of pairs (i < j) with A[i] + A[j] >= M
    T = 0
    j = N - 1
    for i in range(N - 1):
        # Move j down until A[i] + A[j] < M or j == i
        while i < j and A[i] + A[j] >= M:
            j -= 1
        # All indices k with k > j satisfy A[i] + A[k] >= M
        T += (N - j - 1)
    
    # The sum over all pairs of (A_i + A_j) is (N-1) * total_sum.
    # We need to subtract M * T to account for the modulo reduction.
    result = (N - 1) * total_sum - M * T
    print(result)

if __name__ == "__main__":
    main()