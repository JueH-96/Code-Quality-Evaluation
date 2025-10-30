# YOUR CODE HERE
import sys
import threading
import itertools

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Build linear basis over GF(2)
    basis = []
    for num in A:
        for b in basis:
            if num ^ b < num:
                num ^= b
        if num:
            basis.append(num)
    # Normalize basis in descending order
    basis.sort(reverse=True)
    # Remove linear dependencies
    for i in range(len(basis)):
        for j in range(i+1, len(basis)):
            if basis[j] ^ basis[i] < basis[j]:
                basis[j] ^= basis[i]
    
    M = len(basis)
    max_comb = 1
    for i in range(K):
        max_comb = max_comb * (M - i) // (i + 1)
    if max_comb > 1e6:
        # If total combinations exceed limit, and since C(N,K) ≤1e6,
        # K must be ≥ M, so maximum XOR is XOR of basis elements
        max_xor = 0
        for b in basis:
            max_xor ^= b
        print(max_xor)
    else:
        # Generate all combinations of K elements from basis
        max_xor = 0
        for comb in itertools.combinations(basis, K):
            xor_sum = 0
            for num in comb:
                xor_sum ^= num
            if xor_sum > max_xor:
                max_xor = xor_sum
        print(max_xor)

if __name__ == "__main__":
    threading.Thread(target=main).start()