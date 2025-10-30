import sys
from itertools import combinations

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    xor_all = 0
    for num in A:
        xor_all ^= num
    if K == N:
        print(xor_all)
        return

    L = min(K, N - K)
    max_xor = 0
    if K <= N - K:
        for comb in combinations(A, L):
            current_xor = 0
            for num in comb:
                current_xor ^= num
            if current_xor > max_xor:
                max_xor = current_xor
    else:
        for comb in combinations(A, L):
            current_xor = 0
            for num in comb:
                current_xor ^= num
            candidate = xor_all ^ current_xor
            if candidate > max_xor:
                max_xor = candidate
    print(max_xor)

if __name__ == "__main__":
    main()