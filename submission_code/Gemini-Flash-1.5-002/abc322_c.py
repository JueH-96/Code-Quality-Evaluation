# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))

fireworks_days = set(a)

for i in range(1, n + 1):
    days_later = 0
    found = False
    for j in range(i, n + 1):
        if j in fireworks_days:
            days_later = j - i
            found = True
            break
    print(days_later)