N = int(input().strip())
S = [input().strip() for _ in range(N)]

wins = [s.count('o') for s in S]
ranks = sorted(range(N), key=lambda x: -wins[x])

for r in ranks:
    print(r+1, end=' ')