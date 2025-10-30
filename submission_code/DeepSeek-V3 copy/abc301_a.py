# Read input values
N = int(input())
S = input()

# Initialize counters for wins
takahashi_wins = 0
aoki_wins = 0

# Iterate through the string to count wins
for i in range(N):
    if S[i] == 'T':
        takahashi_wins += 1
    else:
        aoki_wins += 1
    # Check if one has already won more than the other
    if takahashi_wins > aoki_wins:
        print('T')
        exit()
    elif aoki_wins > takahashi_wins:
        print('A')
        exit()

# If the loop completes, it means both have the same number of wins
# The winner is the one who reached the final count first
if takahashi_wins > aoki_wins:
    print('T')
else:
    print('A')