# YOUR CODE HERE
N = int(input())

people_bets = []
for i in range(N):
    line = list(map(int, input().split()))
    C_i = line[0]
    bets = line[1:]
    people_bets.append((C_i, set(bets)))

X = int(input())

# Find people who bet on X
winners = []
for i in range(N):
    C_i, bets = people_bets[i]
    if X in bets:
        winners.append((i + 1, C_i))  # (person number, number of bets)

if not winners:
    print(0)
    print()
else:
    # Find minimum number of bets among winners
    min_bets = min(winner[1] for winner in winners)
    
    # Find all winners with minimum bets
    result = []
    for person_num, num_bets in winners:
        if num_bets == min_bets:
            result.append(person_num)
    
    result.sort()
    print(len(result))
    if result:
        print(*result)