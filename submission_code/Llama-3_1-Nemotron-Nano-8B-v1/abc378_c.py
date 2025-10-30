n = int(input())
a = list(map(int, input().split()))
last_occurrence = {}
b = []
for i in range(n):
    num = a[i]
    if num in last_occurrence:
        b.append(str(last_occurrence[num] + 1))
    else:
        b.append('-1')
    last_occurrence[num] = i
print(' '.join(b))