import sys
# But actually, I can use built-in input, no need for sys if I use input()

# Better to use input() for simplicity

# Read N and S using input()
N = int(input())
S = input().strip()  # strip to be safe, though not necessary

# Count the number of wins
tak_count = S.count('T')
aoki_count = S.count('A')

# Compare the total wins
if tak_count > aoki_count:
    print('T')
elif aoki_count > tak_count:
    print('A')
else:
    # Tie situation, find who reached K wins first
    K = tak_count  # or aoki_count, same value
    tak_reach_index = -1
    aoki_reach_index = -1
    tak_cum = 0
    aoki_cum = 0
    for idx in range(N):
        if S[idx] == 'T':
            tak_cum += 1
            if tak_cum == K and tak_reach_index == -1:
                tak_reach_index = idx
        else:  # S[idx] == 'A'
            aoki_cum += 1
            if aoki_cum == K and aoki_reach_index == -1:
                aoki_reach_index = idx
    # Compare the indices to see who reached first
    if tak_reach_index < aoki_reach_index:
        print('T')
    else:
        print('A')