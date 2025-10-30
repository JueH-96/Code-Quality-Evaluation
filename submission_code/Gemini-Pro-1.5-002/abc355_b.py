# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    c = sorted(a + b)
    
    for i in range(n + m - 1):
        if c[i] in a and c[i+1] in a:
            print("Yes")
            return
    
    print("No")

solve()