N = int(input())

people_bets = []
people_counts = []

for i in range(N):
    C_i = int(input())
    bets = list(map(int, input().split()))
    people_counts.append(C_i)
    people_bets.append(set(bets))

X = int(input())

# Find people who bet on X
people_who_bet_on_X = []
for i in range(N):
    if X in people_bets[i]:
        people_who_bet_on_X.append((i+1, people_counts[i]))  # (person_number, count)

if not people_who_bet_on_X:
    print(0)
    print()
else:
    # Find minimum count among those who bet on X
    min_count = min(count for _, count in people_who_bet_on_X)
    
    # Find all people with minimum count
    result = []
    for person, count in people_who_bet_on_X:
        if count == min_count:
            result.append(person)
    
    result.sort()
    print(len(result))
    print(*result)