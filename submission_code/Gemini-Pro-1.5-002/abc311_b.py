# YOUR CODE HERE
def solve():
    n, d = map(int, input().split())
    s = [input() for _ in range(n)]
    
    ans = 0
    for i in range(d):
        for j in range(i, d):
            all_free = True
            for k in range(n):
                for l in range(i, j + 1):
                    if s[k][l] == 'x':
                        all_free = False
                        break
                if not all_free:
                    break
            if all_free:
                ans = max(ans, j - i + 1)
    
    print(ans)

solve()