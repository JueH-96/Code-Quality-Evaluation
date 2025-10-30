import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    M = 10**8

    # Sum of all A_i
    total_sum = sum(A)

    # Count of pairs (i<j) with A[i] + A[j] >= M
    A.sort()
    i = 0
    j = N - 1
    K = 0  # number of pairs with sum >= M
    while i < j:
        if A[i] + A[j] >= M:
            # For this j, all indices from i to j-1 paired with j work
            K += (j - i)
            j -= 1
        else:
            i += 1

    # Sum_{i<j} (A_i + A_j) = (N - 1) * total_sum
    # Each pair where A_i + A_j >= M contributes an extra M that was subtracted by the mod
    # So the final answer is (N-1)*total_sum - M * K
    result = (N - 1) * total_sum - M * K
    print(result)

if __name__ == "__main__":
    main()