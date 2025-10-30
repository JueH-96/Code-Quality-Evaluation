def solve():
    n, q = map(int, input().split())
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().split())))

    dp = {}

    def calculate_dp(index, current_s):
        if index == q:
            return 1

        state = (index, tuple(current_s))
        if state in dp:
            return dp[state]

        p, v = queries[index]
        count = 0

        # Option 1
        valid1 = True
        temp_s1 = list(current_s)
        for i in range(p):
            if temp_s1[i] > v:
                valid1 = False
                break
        if valid1:
            for i in range(p):
                temp_s1[i] = v
            count = (count + calculate_dp(index + 1, temp_s1)) % 998244353

        # Option 2
        valid2 = True
        temp_s2 = list(current_s)
        for i in range(p - 1, n):
            if temp_s2[i] > v:
                valid2 = False
                break

        if valid2:
            for i in range(p - 1, n):
                temp_s2[i] = v
            count = (count + calculate_dp(index + 1, temp_s2)) % 998244353
        
        dp[state] = count
        return count

    initial_s = [0] * n
    result = calculate_dp(0, initial_s)
    print(result)

solve()