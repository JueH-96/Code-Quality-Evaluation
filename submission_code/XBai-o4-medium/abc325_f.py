import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    D = list(map(int, input[ptr:ptr + N]))
    ptr += N
    L1 = int(input[ptr])
    C1 = int(input[ptr+1])
    K1 = int(input[ptr+2])
    ptr += 3
    L2 = int(input[ptr])
    C2 = int(input[ptr+1])
    K2 = int(input[ptr+2])
    ptr += 3

    def generate_options(D_i, L1, L2, C1, C2, max_k1, max_k2):
        options = []
        x_opt = D_i / L1
        x_start = max(0, int(x_opt) - 5)
        x_end = min((D_i + L1 - 1) // L1 + 5, max_k1)
        for x in range(x_start, x_end + 1):
            covered = x * L1
            remaining = D_i - covered
            if remaining <= 0:
                y = 0
            else:
                y = (remaining + L2 - 1) // L2
            if y > max_k2:
                continue
            cost = x * C1 + y * C2
            options.append((x, y, cost))
        return options

    sections = []
    for d in D:
        opts = generate_options(d, L1, L2, C1, C2, K1, K2)
        if not opts:
            print(-1)
            return
        sections.append(opts)

    INF = float('inf')
    dp = [[INF] * (K2 + 1) for _ in range(K1 + 1)]
    dp[0][0] = 0

    for opts in sections:
        new_dp = [[INF] * (K2 + 1) for _ in range(K1 + 1)]
        for k1 in range(K1 + 1):
            for k2 in range(K2 + 1):
                if dp[k1][k2] == INF:
                    continue
                for (x, y, cost) in opts:
                    nk1 = k1 + x
                    nk2 = k2 + y
                    if nk1 > K1 or nk2 > K2:
                        continue
                    if new_dp[nk1][nk2] > dp[k1][k2] + cost:
                        new_dp[nk1][nk2] = dp[k1][k2] + cost
        dp = new_dp

    min_cost = INF
    for k1 in range(K1 + 1):
        for k2 in range(K2 + 1):
            if dp[k1][k2] < min_cost:
                min_cost = dp[k1][k2]
    if min_cost == INF:
        print(-1)
    else:
        print(min_cost)

if __name__ == "__main__":
    main()