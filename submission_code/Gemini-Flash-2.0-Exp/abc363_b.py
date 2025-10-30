N, T, P = map(int, input().split())
L = list(map(int, input().split()))

count = 0
for length in L:
    if length >= T:
        count += 1

if count >= P:
    print(0)
else:
    days = 0
    while True:
        days += 1
        count = 0
        for length in L:
            if length + days >= T:
                count += 1
        if count >= P:
            print(days)
            break