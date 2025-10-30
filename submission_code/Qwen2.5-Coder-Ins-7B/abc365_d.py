# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Initialize the answer to 0
ans = 0

# Iterate over the moves of Aoki
for i in range(N):
    # If Aoki played Rock, Takahashi can play Paper
    if S[i] == 'R':
        ans += 1
    # If Aoki played Paper, Takahashi can play Scissors
    elif S[i] == 'P':
        ans += 1
    # If Aoki played Scissors, Takahashi can play Rock
    else:
        ans += 1

# Print the maximum number of games Takahashi could have won
print(ans)