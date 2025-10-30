n = int(input())
a = list(map(int, input().split()))
result = []
for i in range(n):
    start = i * 7
    end = start + 7
    result.append(sum(a[start:end]))
print(' '.join(map(str, result)))