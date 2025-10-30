import sys
from math import comb

def find_sequence(N, K):
    target = comb(N * K, K, K, K, K, K, K, K, K, K, K, K, K, K, K, K) // 2
    sequence = []
    counts = [K] * N

    for _ in range(N * K):
        for i in range(N):
            if counts[i] > 0:
                temp = counts[:i] + counts[i+1:]
                temp.append(counts[i] - 1)
                current = comb(sum(temp), temp[-1])
                for j in range(len(temp) - 1):
                    current -= comb(sum(temp[:j]), temp[j])
                if current <= target:
                    sequence.append(i + 1)
                    counts[i] -= 1
                    break

    print(*sequence)

def comb(n, *ks):
    result = 1
    for k in ks:
        result *= comb_single(n, k)
        n -= k
    return result

def comb_single(n, k):
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])

    find_sequence(N, K)