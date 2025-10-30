def mod_inverse(a, m):
    """ Return the modular inverse of a under modulo m using Extended Euclidean Algorithm. """
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

def expected_inversion_number(N, K, P):
    MOD = 998244353
    
    # Calculate the initial inversion count
    initial_inversions = 0
    for i in range(N):
        for j in range(i + 1, N):
            if P[i] > P[j]:
                initial_inversions += 1

    # Calculate the expected number of inversions after shuffling
    total_inversions = 0
    for i in range(N - K + 1):
        # Count inversions in the subarray P[i:i+K]
        subarray = P[i:i + K]
        sub_inversions = 0
        for j in range(K):
            for l in range(j + 1, K):
                if subarray[j] > subarray[l]:
                    sub_inversions += 1
        
        # Each shuffle of this subarray contributes to the expected value
        total_inversions += initial_inversions - sub_inversions

    # The expected value is the average over the possible starting positions
    expected_value = (total_inversions + (N - K + 1) * initial_inversions) * mod_inverse(N - K + 1, MOD) % MOD
    
    return expected_value

import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))
    
    result = expected_inversion_number(N, K, P)
    print(result)

if __name__ == "__main__":
    main()