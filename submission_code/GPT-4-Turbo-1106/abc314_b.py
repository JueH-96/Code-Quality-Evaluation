# Read the number of people
N = int(input())

# Initialize a list to store the bets and counts
bets = []

# Read the bets for each person
for i in range(N):
    C_i = int(input())
    A_i = list(map(int, input().split()))
    bets.append((C_i, A_i))

# Read the outcome
X = int(input())

# Find the people who bet on X and have the fewest bets
min_bets = float('inf')
winners = []

for i, (count, numbers) in enumerate(bets, start=1):
    if X in numbers:
        if count < min_bets:
            min_bets = count
            winners = [i]
        elif count == min_bets:
            winners.append(i)

# Print the result
print(len(winners))
if winners:
    print(*winners)