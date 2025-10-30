# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    result = []
    for x in a:
        if x % k == 0:
            result.append(x // k)
            
    result.sort()
    print(*result)

solve()