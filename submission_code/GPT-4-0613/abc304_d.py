import bisect

W, H = map(int, input().split())
N = int(input())
strawberries = [list(map(int, input().split())) for _ in range(N)]
A = int(input())
a = list(map(int, input().split()))
B = int(input())
b = list(map(int, input().split()))

a = [0] + a + [W]
b = [0] + b + [H]

strawberries.sort()

x = [0]*(A+2)
y = [0]*(B+2)

for p, q in strawberries:
    i = bisect.bisect_right(a, p)
    j = bisect.bisect_right(b, q)
    x[i-1] += 1
    y[j-1] += 1

print(min(x), max(x))
print(min(y), max(y))