N, T, P = map(int, input().split())
L = list(map(int, input().split()))
L.sort()
days = 0
while len([l for l in L if l >= T]) < P:
    L = [l+1 for l in L]
    days += 1
print(days)