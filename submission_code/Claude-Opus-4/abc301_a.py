# YOUR CODE HERE
N = int(input())
S = input()

# Count total wins
t_wins = S.count('T')
a_wins = S.count('A')

# If different number of wins, the one with more wins is the winner
if t_wins > a_wins:
    print('T')
elif a_wins > t_wins:
    print('A')
else:
    # Same number of wins, find who reached that number first
    target_wins = t_wins  # Both have the same number
    t_count = 0
    a_count = 0
    
    for i in range(N):
        if S[i] == 'T':
            t_count += 1
            if t_count == target_wins:
                print('T')
                break
        else:  # S[i] == 'A'
            a_count += 1
            if a_count == target_wins:
                print('A')
                break