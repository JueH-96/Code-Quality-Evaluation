n = int(input())
a = list(map(int, input().split()))
sums = [sum(a[i*7 : i*7 +7]) for i in range(n)]
print(' '.join(map(str, sums)))