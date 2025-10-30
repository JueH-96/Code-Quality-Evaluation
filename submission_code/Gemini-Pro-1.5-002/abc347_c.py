# YOUR CODE HERE
def solve():
    n, a, b = map(int, input().split())
    d = list(map(int, input().split()))
    
    for start in range(a + b):
        all_holidays = True
        for di in d:
            day = (start + di) % (a + b)
            if day >= a:
                all_holidays = False
                break
        if all_holidays:
            print("Yes")
            return
    
    print("No")

solve()