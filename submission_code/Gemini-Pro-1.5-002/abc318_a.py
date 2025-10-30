# YOUR CODE HERE
n, m, p = map(int, input().split())

count = 0
day = m
while day <= n:
    count += 1
    day += p

print(count)