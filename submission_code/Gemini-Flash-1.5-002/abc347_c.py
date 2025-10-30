# YOUR CODE HERE
N, A, B = map(int, input().split())
D = list(map(int, input().split()))

possible = False
for start_day in range(A + B):
    all_holidays = True
    for d in D:
        day = (start_day + d) % (A + B)
        if day >= A:
            all_holidays = False
            break
    if all_holidays:
        possible = True
        break

if possible:
    print("Yes")
else:
    print("No")