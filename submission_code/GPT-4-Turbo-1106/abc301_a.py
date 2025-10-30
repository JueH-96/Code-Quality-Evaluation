# Read input
N = int(input().strip())
S = input().strip()

# Initialize counters
takahashi_wins = 0
aoki_wins = 0

# Process the results
for i in range(N):
    if S[i] == 'T':
        takahashi_wins += 1
    else:
        aoki_wins += 1
    
    # Check if someone has already won more games
    if takahashi_wins > N // 2 or aoki_wins > N // 2:
        break

# Determine the winner
if takahashi_wins > aoki_wins:
    print('T')
elif aoki_wins > takahashi_wins:
    print('A')
else:
    # If the number of wins is the same, find who reached the number first
    for result in S:
        if result == 'T':
            print('T')
            break
        elif result == 'A':
            print('A')
            break