# Read N
N = int(input())

# Store information for each person
people = []
for i in range(N):
    C_i = int(input())
    bets = list(map(int, input().split()))
    people.append((C_i, bets))

# Read X
X = int(input())

# Find people who bet on X
people_who_bet_on_X = []
for i in range(N):
    C_i, bets = people[i]
    if X in bets:
        people_who_bet_on_X.append((i + 1, C_i))  # i+1 because people are 1-indexed

# If no one bet on X, output 0 and empty line
if not people_who_bet_on_X:
    print(0)
    print()
else:
    # Find minimum C_i among people who bet on X
    min_bets = min(C_i for _, C_i in people_who_bet_on_X)
    
    # Find all people with minimum bets
    result = [person for person, C_i in people_who_bet_on_X if C_i == min_bets]
    result.sort()  # Ensure ascending order
    
    # Output
    print(len(result))
    print(' '.join(map(str, result)))