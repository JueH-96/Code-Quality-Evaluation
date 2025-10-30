# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    s = data[1]
    
    ans = 0
    prefix = 0
    for i in range(n):
        prefix += int(s[i])
        ans += prefix * (n - i)
    
    print(ans)
solve()