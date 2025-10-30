def solve():
    n = int(input())
    a = list(map(int, input().split()))

    dp = {}

    def get_dp(index, count):
        if index == n:
            return 0
        if (index, count) in dp:
            return dp[(index, count)]

        res = get_dp(index + 1, count)
        
        if count % 2 == 0:
            res = max(res, get_dp(index + 1, count + 1) + a[index] * 2)
        else:
            res = max(res, get_dp(index + 1, count + 1) + a[index])

        dp[(index, count)] = res
        return res

    print(get_dp(0, 0))

solve()