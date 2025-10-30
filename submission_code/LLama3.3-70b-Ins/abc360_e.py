import sys

def read_input():
    return sys.stdin.readline().strip()

def read_ints():
    return list(map(int, read_input().split()))

def solve():
    N, K = read_ints()
    MOD = 998244353
    prob = [0] * N
    prob[0] = 1
    for _ in range(K):
        new_prob = [0] * N
        for i in range(N):
            for j in range(N):
                if i != j:
                    new_prob[j] += prob[i] * (1 / N) * (1 / N)
                    new_prob[i] += prob[i] * (1 / N) * (1 / N)
        prob = new_prob
    expected_value = sum(i * p for i, p in enumerate(prob, start=1))
    expected_value = int(expected_value * pow(N * N, MOD - 2, MOD)) % MOD
    print(expected_value)

if __name__ == "__main__":
    solve()