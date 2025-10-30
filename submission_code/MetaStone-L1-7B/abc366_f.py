import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    K = int(data[idx+1])
    idx += 2
    functions = []
    for _ in range(N):
        a = int(data[idx])
        b = int(data[idx+1])
        functions.append((a, b))
        idx += 2

    # Initialize DP: each function is a possible starting point
    dp = {i: (a, b) for i, (a, b) in enumerate(functions)}

    for step in range(2, K + 1):
        next_dp = {}
        for f in dp:
            a_f, b_f = dp[f]
            for g in range(len(functions)):
                if g == f:
                    continue
                a_g, b_g = functions[g]
                new_a = a_f * a_g
                new_b = b_f * a_g + b_g
                if g in next_dp:
                    current = next_dp[g][0] + next_dp[g][1]
                    if current < new_a + new_b:
                        next_dp[g] = (new_a, new_b)
                else:
                    next_dp[g] = (new_a, new_b)
        dp = next_dp

    max_val = 0
    for a, b in dp.values():
        current = a + b
        if current > max_val:
            max_val = current
    print(max_val)

if __name__ == '__main__':
    main()