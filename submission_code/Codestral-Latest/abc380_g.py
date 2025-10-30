import sys
from itertools import permutations

MOD = 998244353

def inversion_number(perm):
    n = len(perm)
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv_count += 1
    return inv_count

def expected_inversion_number(N, K, P):
    total_inversions = 0
    total_probability = 0

    for i in range(1, N - K + 2):
        for perm in permutations(P[i-1:i+K-1]):
            new_perm = P[:i-1] + list(perm) + P[i+K-1:]
            inv_count = inversion_number(new_perm)
            total_inversions += inv_count
            total_probability += 1

    expected_value = total_inversions / total_probability
    expected_value %= MOD
    return int(expected_value)

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))

    result = expected_inversion_number(N, K, P)
    print(result)

if __name__ == "__main__":
    main()