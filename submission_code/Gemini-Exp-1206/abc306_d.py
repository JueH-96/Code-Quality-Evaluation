def solve():
    n = int(input())
    courses = []
    for _ in range(n):
        x, y = map(int, input().split())
        courses.append((x, y))

    dp = {}

    def get_dp(index, stomach):
        if index == n:
            return 0
        if (index, stomach) in dp:
            return dp[(index, stomach)]

        x, y = courses[index]
        
        ans = get_dp(index + 1, stomach) 

        if stomach == 0:
            if x == 0:
                ans = max(ans, y + get_dp(index + 1, 0))
            else:
                ans = max(ans, y + get_dp(index + 1, 1))
        else:
            if x == 0:
                ans = max(ans, y + get_dp(index + 1, 0))
            
        dp[(index, stomach)] = ans
        return ans

    print(get_dp(0, 0))

solve()