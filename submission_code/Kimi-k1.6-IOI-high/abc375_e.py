def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    people = []
    total = 0
    for _ in range(N):
        A = int(input[idx])
        B = int(input[idx+1])
        people.append((A, B))
        total += B
        idx += 2
    if total % 3 != 0:
        print(-1)
        return
    target = total // 3
    dp = {(0, 0, 0): 0}
    for a, b in people:
        next_dp = {}
        for (s1, s2, s3), changes in dp.items():
            for t in [1, 2, 3]:
                new_s1 = s1 + (b if t == 1 else 0)
                new_s2 = s2 + (b if t == 2 else 0)
                new_s3 = s3 + (b if t == 3 else 0)
                if new_s1 > target or new_s2 > target or new_s3 > target:
                    continue
                new_changes = changes + (1 if t != a else 0)
                key = (new_s1, new_s2, new_s3)
                if key in next_dp:
                    if new_changes < next_dp[key]:
                        next_dp[key] = new_changes
                else:
                    next_dp[key] = new_changes
        dp = next_dp
    if (target, target, target) in dp:
        print(dp[(target, target, target)])
    else:
        print(-1)

if __name__ == '__main__':
    main()