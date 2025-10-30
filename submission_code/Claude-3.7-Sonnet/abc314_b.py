# Read the number of people
N = int(input())

# Store the bets of each person
bets = []
for i in range(N):
    C_i = int(input())
    A_i = list(map(int, input().split()))
    bets.append((C_i, A_i))
    
# Read the outcome of the spin
X = int(input())

# Find people who have bet on X and the minimum number of bets
min_bets = float('inf')
winners = []

for i in range(N):
    C_i, A_i = bets[i]
    if X in A_i:
        if C_i < min_bets:
            min_bets = C_i
            winners = [i + 1]  # Add 1 because people are numbered from 1
        elif C_i == min_bets:
            winners.append(i + 1)

# Print the results
print(len(winners))
if winners:
    print(' '.join(map(str, winners)))