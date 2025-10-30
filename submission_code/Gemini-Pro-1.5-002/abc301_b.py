# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    while True:
        inserted = False
        for i in range(len(a) - 1):
            if abs(a[i] - a[i+1]) != 1:
                if a[i] < a[i+1]:
                    insertion = list(range(a[i] + 1, a[i+1]))
                else:
                    insertion = list(range(a[i] - 1, a[i+1], -1))
                
                a = a[:i+1] + insertion + a[i+1:]
                inserted = True
                break
        
        if not inserted:
            break
    
    print(*a)

solve()