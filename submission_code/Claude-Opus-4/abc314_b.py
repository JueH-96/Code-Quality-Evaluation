# YOUR CODE HERE
N = int(input())

people_bets = []
for i in range(N):
    C_i = int(input())
    bets = list(map(int, input().split()))
    people_bets.append((C_i, bets))

X = int(input())

# Find people who bet on X and their bet counts
people_who_bet_on_X = []
for i in range(N):
    C_i, bets = people_bets[i]
    if X in bets:
        people_who_bet_on_X.append((i + 1, C_i))  # (person number, bet count)

# If no one bet on X
if not people_who_bet_on_X:
    print(0)
    print()
else:
    # Find minimum bet count among those who bet on X
    min_bet_count = min(bet_count for _, bet_count in people_who_bet_on_X)
    
    # Find all people with minimum bet count
    result = []
    for person_num, bet_count in people_who_bet_on_X:
        if bet_count == min_bet_count:
            result.append(person_num)
    
    # Output
    print(len(result))
    print(' '.join(map(str, result)))