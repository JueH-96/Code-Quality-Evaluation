# YOUR CODE HERE
def solve():
    d = int(input())
    ans = float('inf')
    for x in range(int(d**0.5) + 2):
        x2 = x * x
        if x2 > d:
            break
        y2_ideal = d - x2
        y = int(y2_ideal**0.5)
        
        ans = min(ans, abs(x2 + y*y - d))
        y += 1
        ans = min(ans, abs(x2 + y*y - d))
        
    print(ans)

solve()