# YOUR CODE HERE
import sys
from itertools import combinations

MOD = 998244353

def modinv(a, p):
    # Fermat's little theorem: a^(p-1) ≡ 1 (mod p) implies a^(p-2) ≡ a^(-1) (mod p)
    return pow(a, p - 2, p)

def count_inversions(arr):
    inversions = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions

def expected_inversions(N, K, P):
    total_inversions = 0
    num_positions = N - K + 1
    
    for start in range(num_positions):
        subarray = P[start:start + K]
        subarray_inversions = 0
        num_permutations = 0
        
        # Calculate inversions for each permutation of the subarray
        for perm in combinations(range(K), 2):
            if subarray[perm[0]] > subarray[perm[1]]:
                subarray_inversions += 1
            num_permutations += 1
        
        # Average inversions for this subarray
        total_inversions += subarray_inversions
    
    # Calculate expected value
    expected_value = total_inversions * modinv(num_positions * num_permutations, MOD) % MOD
    return expected_value

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))
    
    result = expected_inversions(N, K, P)
    print(result)

if __name__ == "__main__":
    main()