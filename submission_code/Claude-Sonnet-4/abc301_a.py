# YOUR CODE HERE
N = int(input())
S = input().strip()

# Count total wins
t_wins = S.count('T')
a_wins = S.count('A')

# If one player has more wins, they're the overall winner
if t_wins > a_wins:
    print('T')
elif a_wins > t_wins:
    print('A')
else:
    # Equal wins - find who reached the target first
    target = t_wins  # Same as a_wins since they're equal
    t_count = 0
    a_count = 0
    
    for char in S:
        if char == 'T':
            t_count += 1
        else:
            a_count += 1
        
        # Check if either player reached the target
        if t_count == target:
            print('T')
            break
        elif a_count == target:
            print('A')
            break