N = int(input())
bets = []
for i in range(N):
    C = int(input())
    A = list(map(int, input().split()))
    bets.append(A)
X = int(input())

# Find people who bet on X and their bet counts
winners = []
for i in range(N):
    if X in bets[i]:
        winners.append((len(bets[i]), i+1))

if not winners:
    print(0)
else:
    # Find minimum number of bets among winners
    min_bets = min(w[0] for w in winners)
    
    # Get people with minimum bets who won
    result = [w[1] for w in winners if w[0] == min_bets]
    result.sort()
    
    print(len(result))
    print(*result)