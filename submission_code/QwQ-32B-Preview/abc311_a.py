N = int(input())
S = input().strip()

seen = set()
for i in range(N):
    char = S[i]
    seen.add(char)
    if len(seen) == 3:
        print(i + 1)
        break