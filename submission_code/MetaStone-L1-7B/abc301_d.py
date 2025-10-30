S = input().strip()
N = int(input())

current = 0
for i in range(len(S)):
    c = S[i]
    if c == '0' or c == '1':
        current = current * 2 + (1 if c == '1' else 0)
    else:
        max_possible = (current << 1) + ((1 << (len(S) - i - 1)) - 1)
        if max_possible <= N:
            current = current << 1 | 1
        else:
            current = current << 1

if current <= N:
    print(current)
else:
    print(-1)