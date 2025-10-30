import sys
from collections import deque

MOD = 998244353

def mod_inverse(a, m):
    return pow(a, m - 2, m)

def solve():
    input = sys.stdin.read
    data = input().split()

    index = 0
    T = int(data[index])
    index += 1
    results = []

    for _ in range(T):
        N = int(data[index])
        index += 1
        p = list(map(int, data[index:index + N]))
        index += N
        a = list(map(int, data[index:index + N]))
        index += N

        total_prob = sum(a)
        probs = [x / total_prob for x in a]

        # Calculate the expected number of operations
        expected_ops = 0
        for i in range(N):
            expected_ops += probs[i] * (N - i)

        # Convert the expected value to the required format
        expected_ops *= mod_inverse(total_prob, MOD)
        expected_ops %= MOD

        results.append(int(expected_ops))

    for result in results:
        print(result)

if __name__ == "__main__":
    solve()