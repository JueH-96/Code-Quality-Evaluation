# YOUR CODE HERE
n, t, p = map(int, input().split())
l = list(map(int, input().split()))

count = 0
for hair_length in l:
    if hair_length >= t:
        count += 1

if count >= p:
    print(0)
else:
    days = 0
    while count < p:
        days += 1
        for i in range(n):
            if l[i] + days >= t and l[i] + days -1 < t:
                count += 1
    print(days)