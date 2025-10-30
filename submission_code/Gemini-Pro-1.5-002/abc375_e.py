def solve():
    n = int(input())
    ab = []
    for _ in range(n):
        a, b = map(int, input().split())
        ab.append((a, b))

    total_strength = sum(b for _, b in ab)
    if total_strength % 3 != 0:
        print(-1)
        return

    target_strength = total_strength // 3

    dp = {}

    def get_dp(idx, s1, s2, s3):
        if idx == n:
            if s1 == target_strength and s2 == target_strength and s3 == target_strength:
                return 0
            else:
                return float('inf')

        if (idx, s1, s2, s3) in dp:
            return dp[(idx, s1, s2, s3)]
        
        a, b = ab[idx]
        
        res = float('inf')
        
        # Keep in original team
        if a == 1:
            if s1 + b <= target_strength:
                res = min(res, get_dp(idx + 1, s1 + b, s2, s3))
        elif a == 2:
            if s2 + b <= target_strength:
                res = min(res, get_dp(idx + 1, s1, s2 + b, s3))
        else:
            if s3 + b <= target_strength:
                res = min(res, get_dp(idx + 1, s1, s2, s3 + b))

        # Switch to team 1
        if a != 1:
            if s1 + b <= target_strength:
                res = min(res, 1 + get_dp(idx + 1, s1 + b, s2, s3))
        
        # Switch to team 2
        if a != 2:
            if s2 + b <= target_strength:
                res = min(res, 1 + get_dp(idx + 1, s1, s2 + b, s3))

        # Switch to team 3
        if a != 3:
            if s3 + b <= target_strength:
                res = min(res, 1 + get_dp(idx + 1, s1, s2, s3 + b))

        dp[(idx, s1, s2, s3)] = res
        return res

    ans = get_dp(0, 0, 0, 0)
    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

solve()