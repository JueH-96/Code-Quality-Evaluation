import sys
data = sys.stdin.read().split()
N = int(data[0])
S = data[1]

# Set initial dp_prev for game 0
a0 = S[0]
if a0 == 'R':
    dp_prev = [0, 1, -1]  # R, P, S indices
elif a0 == 'P':
    dp_prev = [-1, 0, 1]
elif a0 == 'S':
    dp_prev = [1, -1, 0]

# Now for each subsequent game from 1 to N-1
for i in range(1, N):
    a_curr = S[i]
    dp_curr = [-1, -1, -1]
    for j in range(3):  # current move: 0=R, 1=P, 2=S
        # Check if valid for a_curr
        if (a_curr == 'R' and j in (0, 1)) or \
           (a_curr == 'P' and j in (1, 2)) or \
           (a_curr == 'S' and j in (0, 2)):
            # Valid, compute bonus
            if (j == 0 and a_curr == 'S') or \
               (j == 1 and a_curr == 'R') or \
               (j == 2 and a_curr == 'P'):
                bonus = 1
            else:
                bonus = 0
            # Now options
            options = [dp_prev[k] + bonus for k in range(3) if k != j and dp_prev[k] != -1]
            if options:
                dp_curr[j] = max(options)
            else:
                dp_curr[j] = -1
        # else already -1
    # Set dp_prev to dp_curr copy
    dp_prev = dp_curr[:]

# After all, find max in dp_prev
ans = max(dp_prev)
print(ans)