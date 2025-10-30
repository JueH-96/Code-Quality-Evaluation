import sys
from itertools import combinations

MOD = 998244353

def is_arithmetic(seq):
    if len(seq) <= 1:
        return True
    diff = seq[1] - seq[0]
    for i in range(2, len(seq)):
        if seq[i] - seq[i-1] != diff:
            return False
    return True

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    result = [0] * N

    for k in range(1, N + 1):
        count = 0
        for comb in combinations(A, k):
            if is_arithmetic(comb):
                count += 1
        result[k-1] = count % MOD

    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()