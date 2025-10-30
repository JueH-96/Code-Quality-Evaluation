n = int(input())
a = list(map(int, input().split()))
b = [a[i] * a[i + 1] for i in range(n - 1)]
print(' '.join(map(str, b)))