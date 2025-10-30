from itertools import combinations
from collections import Counter
from math import comb

MOD = 998244353

def is_arithmetic(seq):
    if len(seq) < 2:
        return True
    diff = seq[1] - seq[0]
    return all(seq[i+1] - seq[i] == diff for i in range(1, len(seq)))

def count_arithmetic_subsequences(arr):
    n = len(arr)
    result = [0] * n
    for k in range(1, n+1):
        for indices in combinations(range(n), k):
            subseq = [arr[i] for i in indices]
            if is_arithmetic(subseq):
                result[k-1] += 1
        result[k-1] %= MOD
    return result

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    result = count_arithmetic_subsequences(arr)
    print(*result)

if __name__ == "__main__":
    main()