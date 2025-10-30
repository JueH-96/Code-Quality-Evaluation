N = int(input())
S = input().strip()

total = 0
for i in range(N):
    current = int(S[i])
    total += current
    for j in range(i + 1, N):
        if current == 1:
            current = 1 - int(S[j])
        else:
            current = 1
        total += current

print(total)