N, M, P = map(int, input().split())

count = 0
day = M
while day <= N:
    count += 1
    day += P

print(count)