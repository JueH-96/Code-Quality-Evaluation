import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    queries = [(int(data[3 + 2 * i]), int(data[4 + 2 * i])) for i in range(Q)]

    # Precompute prefix counts
    prefix_1 = [0] * (N + 1)
    prefix_2 = [0] * (N + 1)
    prefix_slash = [0] * (N + 1)

    for i in range(1, N + 1):
        prefix_1[i] = prefix_1[i - 1] + (1 if S[i - 1] == '1' else 0)
        prefix_2[i] = prefix_2[i - 1] + (1 if S[i - 1] == '2' else 0)
        prefix_slash[i] = prefix_slash[i - 1] + (1 if S[i - 1] == '/' else 0)

    results = []

    for L, R in queries:
        count_1 = prefix_1[R] - prefix_1[L - 1]
        count_2 = prefix_2[R] - prefix_2[L - 1]
        count_slash = prefix_slash[R] - prefix_slash[L - 1]

        if count_slash == 0:
            results.append(0)
            continue

        max_len = 1
        for i in range(1, min(count_1, count_2) + 1):
            if i <= count_slash:
                max_len = max(max_len, 2 * i + 1)

        results.append(max_len)

    sys.stdout.write('
'.join(map(str, results)) + '
')

solve()