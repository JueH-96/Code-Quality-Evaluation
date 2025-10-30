# YOUR CODE HERE
def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    
    ans = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            count = 0
            for k in range(min(len(s[i]), len(s[j]))):
                if s[i][k] == s[j][k]:
                    count += 1
                else:
                    break
            ans += count
    print(ans)

solve()