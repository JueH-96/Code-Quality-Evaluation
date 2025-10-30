N = int(input())
pairs = [input().split() for _ in range(N)]
ratings = [int(C) for _, C in pairs]
lex_sorted = sorted(enumerate(pairs), key=lambda x: x[1][0])

T = sum(ratings)
winner = T % N

for i, _ in lex_sorted:
    if i == winner:
        print(pairs[i][0])
        break