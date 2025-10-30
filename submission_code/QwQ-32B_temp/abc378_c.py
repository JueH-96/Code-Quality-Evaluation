n = int(input())
A = list(map(int, input().split()))
last_occurrence = {}
B = []
for idx in range(n):
    num = A[idx]
    pos = idx + 1
    if num in last_occurrence:
        B.append(last_occurrence[num])
    else:
        B.append(-1)
    last_occurrence[num] = pos
print(' '.join(map(str, B)))