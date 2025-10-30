import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Initialize index to keep track of position in data
idx = 0

# Read N
N = int(data[idx])
idx += 1

# Store information for each person: number of bets and set of bets
people = []
for _ in range(N):
    C = int(data[idx])  # Read C_i
    idx += 1
    bets = set()  # Use a set for fast lookup of bets
    for _ in range(C):
        bet = int(data[idx])  # Read each bet
        bets.add(bet)
        idx += 1
    people.append((C, bets))  # Store (C_i, set of bets)

# Read X, the outcome
X = int(data[idx])

# Find indices of people who bet on X (0-based indices)
candidates = [i for i in range(N) if X in people[i][1]]

# If no one bet on X, output 0
if not candidates:
    print(0)
else:
    # Find the minimum number of bets among those who bet on X
    min_C = min(people[i][0] for i in candidates)
    
    # Find indices of people who bet on X and have the minimum number of bets
    selected = [i for i in candidates if people[i][0] == min_C]
    
    # Get person numbers (1-based) and sort them (though they should be in order)
    person_nums = [i + 1 for i in selected]
    person_nums.sort()  # Ensure sorted in ascending order
    
    # Get the count K
    K = len(person_nums)
    
    # Output K on the first line
    print(K)
    
    # Output the person numbers separated by spaces on the second line
    print(' '.join(map(str, person_nums)))