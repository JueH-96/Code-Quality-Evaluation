import sys
import math

MOD = 998244353

def inv_mod(x):
    return pow(x, MOD - 2, MOD)

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))

    # Calculate the initial inversion count
    inv_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if P[i] > P[j]:
                inv_count += 1

    # Calculate the expected value of the inversion count after the operation
    expected_value = 0
    for i in range(1, N - K + 2):
        # Calculate the number of inversions in the subarray P[i:i+K]
        subarray_inv_count = 0
        for j in range(i, i + K):
            for k in range(j + 1, i + K):
                if P[j] > P[k]:
                    subarray_inv_count += 1

        # Calculate the expected number of inversions in the subarray after shuffling
        expected_subarray_inv_count = subarray_inv_count * inv_mod(math.comb(K, 2))

        # Add the expected number of inversions in the subarray to the total expected value
        expected_value = (expected_value + expected_subarray_inv_count) % MOD

    # Calculate the final expected value
    expected_value = (inv_count + expected_value * inv_mod(N - K + 1)) % MOD

    # Print the result
    print(expected_value)

if __name__ == "__main__":
    main()