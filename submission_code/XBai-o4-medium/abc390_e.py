import sys

def compute_dp(foods, max_cal):
    dp = [0] * (max_cal + 1)
    for a_i, c_i in foods:
        for j in range(max_cal, c_i - 1, -1):
            if dp[j - c_i] + a_i > dp[j]:
                dp[j] = dp[j - c_i] + a_i
    return dp

def find_min_cal(dp, m, max_cal):
    if dp[-1] < m:
        return max_cal + 1
    left = 0
    right = max_cal
    res = max_cal + 1
    while left <= right:
        c = (left + right) // 2
        if dp[c] >= m:
            res = c
            right = c - 1
        else:
            left = c + 1
    return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    vit1 = []
    vit2 = []
    vit3 = []
    for _ in range(N):
        v = int(input[ptr])
        ptr += 1
        a = int(input[ptr])
        ptr += 1
        c = int(input[ptr])
        ptr += 1
        if v == 1:
            vit1.append( (a, c) )
        elif v == 2:
            vit2.append( (a, c) )
        else:
            vit3.append( (a, c) )
    dp1 = compute_dp(vit1, X)
    dp2 = compute_dp(vit2, X)
    dp3 = compute_dp(vit3, X)
    sum_v1 = dp1[-1]
    sum_v2 = dp2[-1]
    sum_v3 = dp3[-1]
    upper = min(sum_v1, sum_v2, sum_v3)
    low = 0
    high = upper
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        c1 = find_min_cal(dp1, mid, X)
        c2 = find_min_cal(dp2, mid, X)
        c3 = find_min_cal(dp3, mid, X)
        if c1 + c2 + c3 <= X:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    print(ans)

if __name__ == '__main__':
    main()