# Read the input
n = int(input())
s = input()

# Initialize the scores for Takahashi and Aoki
takahashi_score = 0
aoki_score = 0

# Iterate through the games and update the scores
for i in range(n):
    if s[i] == 'T':
        takahashi_score += 1
    else:
        aoki_score += 1

# Determine the overall winner
if takahashi_score > aoki_score:
    print('T')
elif aoki_score > takahashi_score:
    print('A')
else:
    # Check who reached the same number of wins first
    for i in range(n):
        if s[i] == 'T':
            takahashi_score += 1
            if takahashi_score == aoki_score:
                print('T')
                break
        else:
            aoki_score += 1
            if takahashi_score == aoki_score:
                print('A')
                break