import sys
from itertools import chain, combinations

def powerset(s):
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

def xor(lst):
    result = 0
    for num in lst:
        result ^= num
    return result

def calculate_score(subsequence, M, K):
    if len(subsequence) % M == 0:
        return pow(xor(subsequence), K, 998244353)
    else:
        return 0

def main():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))

    total_score = 0
    for subsequence in powerset(A):
        total_score += calculate_score(subsequence, M, K)
        total_score %= 998244353

    print(total_score)

if __name__ == "__main__":
    main()