N = int(input())
rates = []
for i in range(N):
    A, B = map(int, input().split())
    rates.append((A / (A + B), i + 1))
rates.sort(key=lambda x: (-x[0], x[1]))
for _, i in rates:
    print(i)