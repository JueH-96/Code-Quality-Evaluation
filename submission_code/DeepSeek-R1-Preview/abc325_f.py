def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    D = list(map(int, input[ptr:ptr+N]))
    ptr += N
    L1, C1, K1 = map(int, input[ptr:ptr+3])
    ptr +=3
    L2, C2, K2 = map(int, input[ptr:ptr+3])
    ptr +=3

    sections = []
    for d in D:
        options = []
        x1_max = min((d + L1 - 1) // L1, K1)
        for x1 in range(0, x1_max + 1):
            remaining = d - x1 * L1
            if remaining <= 0:
                x2 = 0
            else:
                x2 = (remaining + L2 - 1) // L2
            if x2 <= K2:
                options.append((x1, x2))
        sections.append(options)

    current_dp = {(0, 0): 0}
    for options in sections:
        new_dp = {}
        for (a, b), cost in current_dp.items():
            for x1, x2 in options:
                new_a = a + x1
                new_b = b + x2
                if new_a > K1 or new_b > K2:
                    continue
                new_cost = cost + x1 * C1 + x2 * C2
                key = (new_a, new_b)
                if key in new_dp:
                    if new_cost < new_dp[key]:
                        new_dp[key] = new_cost
                else:
                    new_dp[key] = new_cost
        current_dp = new_dp
        if not current_dp:
            break

    if current_dp:
        min_cost = min(current_dp.values())
        print(min_cost)
    else:
        print(-1)

if __name__ == '__main__':
    main()