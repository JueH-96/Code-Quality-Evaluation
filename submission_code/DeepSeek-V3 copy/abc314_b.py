# Read the number of people
N = int(input())

# Initialize a list to store the bets of each person
bets = []

# Read the bets for each person
for i in range(N):
    C_i = int(input())
    A_i = list(map(int, input().split()))
    bets.append((C_i, A_i))

# Read the outcome of the roulette spin
X = int(input())

# Find all people who have bet on X
candidates = []
for idx, (C_i, A_i) in enumerate(bets, start=1):
    if X in A_i:
        candidates.append((C_i, idx))

# If no one bet on X, print 0 and exit
if not candidates:
    print(0)
else:
    # Find the minimum number of bets among the candidates
    min_bets = min(candidate[0] for candidate in candidates)
    
    # Find all candidates with the minimum number of bets
    result = [candidate[1] for candidate in candidates if candidate[0] == min_bets]
    
    # Print the count and the sorted list of indices
    print(len(result))
    print(' '.join(map(str, sorted(result))))