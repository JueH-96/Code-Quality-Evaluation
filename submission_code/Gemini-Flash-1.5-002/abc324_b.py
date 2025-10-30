# YOUR CODE HERE
def solve():
    n = int(input())
    
    x = 0
    while n % 2 == 0:
        n //= 2
        x += 1
    
    y = 0
    while n % 3 == 0:
        n //= 3
        y += 1
    
    if n == 1:
        print("Yes")
    else:
        print("No")

solve()