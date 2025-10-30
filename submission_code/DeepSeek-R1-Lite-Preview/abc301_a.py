# Read input
N = int(input())
S = input()

# Count total wins
t_count = S.count('T')
a_count = S.count('A')

if t_count != a_count:
    # The player with more wins is the overall winner
    if t_count > a_count:
        print('T')
    else:
        print('A')
else:
    # Both have the same number of wins, find who reached it first
    t_reach = -1
    a_reach = -1
    t_current = 0
    a_current = 0
    for i in range(N):
        if S[i] == 'T':
            t_current += 1
            if t_current == t_count and t_reach == -1:
                t_reach = i + 1  # 1-based indexing
        if S[i] == 'A':
            a_current += 1
            if a_current == a_count and a_reach == -1:
                a_reach = i + 1
    if t_reach < a_reach:
        print('T')
    else:
        print('A')