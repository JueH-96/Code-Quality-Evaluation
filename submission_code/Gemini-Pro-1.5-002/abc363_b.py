# YOUR CODE HERE
def solve():
    n, t, p = map(int, input().split())
    l = list(map(int, input().split()))

    count = 0
    for length in l:
        if length >= t:
            count += 1
    
    if count >= p:
        print(0)
        return

    days = 0
    while True:
        days += 1
        count = 0
        for length in l:
            if length + days >= t:
                count += 1
        if count >= p:
            print(days)
            return

solve()