# YOUR CODE HERE
n = int(input())
bets = []
for _ in range(n):
    c = int(input())
    bet_set = set(map(int, input().split()))
    if int(input()) in bet_set:
        bets.append((c, _ + 1))
min_bets = min(b[0] for b in bets)
result = [b[1] for b in bets if b[0] == min_bets]
print(len(result))
print(' '.join(map(str, result)))