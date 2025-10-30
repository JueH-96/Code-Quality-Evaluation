N, M, P = map(int, input().split())

days = 0
while M <= N:
    M += P
    days += 1

print(days)