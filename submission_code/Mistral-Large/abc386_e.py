import sys
from itertools import combinations

def max_xor_k_elements(N, K, A):
    max_xor = 0
    for comb in combinations(A, K):
        current_xor = 0
        for num in comb:
            current_xor ^= num
        if current_xor > max_xor:
            max_xor = current_xor
    return max_xor

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    result = max_xor_k_elements(N, K, A)
    print(result)

if __name__ == "__main__":
    main()