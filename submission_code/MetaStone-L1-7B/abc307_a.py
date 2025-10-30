n = int(input())
a = list(map(int, input().split()))
result = []
for i in range(n):
    start = i * 7
    end = start + 6
    total = sum(a[start:end+1])
    result.append(str(total))
print(' '.join(result))