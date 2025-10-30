N = int(input())
S = input().strip()

seen = set()
for i in range(N):
    seen.add(S[i])
    if len(seen) == 3:
        print(i + 1)
        break