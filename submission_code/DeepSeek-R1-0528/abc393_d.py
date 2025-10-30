n = int(input().strip())
s = input().strip()

ones = [i for i, char in enumerate(s) if char == '1']
k = len(ones)

if k == 0:
    print(0)
else:
    q = [ones[i] - i for i in range(k)]
    q.sort()
    mid = k // 2
    total = sum(q[-mid:]) - sum(q[:mid])
    print(total)