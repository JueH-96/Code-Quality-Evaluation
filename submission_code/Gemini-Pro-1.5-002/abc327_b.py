# YOUR CODE HERE
def solve():
    b = int(input())
    
    if b == 1:
        print(1)
        return

    for a in range(2, 40):
        if a**a == b:
            print(a)
            return
    
    print(-1)

solve()