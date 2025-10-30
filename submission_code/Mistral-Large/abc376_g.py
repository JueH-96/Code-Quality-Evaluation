import sys
input = sys.stdin.read
MOD = 998244353

def main():
    T, *cases = map(int, input().split())
    index = 0
    results = []

    for _ in range(T):
        N = cases[index]
        index += 1
        p = cases[index:index + N]
        index += N
        a = cases[index:index + N]
        index += N

        total_a = sum(a)
        expected_value = 0

        for i in range(N):
            expected_value += a[i] * (i + 1)
            expected_value %= MOD

        expected_value *= pow(total_a, MOD - 2, MOD)
        expected_value %= MOD

        results.append(expected_value)

    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()