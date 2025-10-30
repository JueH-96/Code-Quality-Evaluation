N = int(input())
S = input()

t_wins = S.count('T')
a_wins = S.count('A')

if t_wins > a_wins:
    print('T')
elif a_wins > t_wins:
    print('A')
else:
    # Tie case: find who reached the final win count first
    target_wins = t_wins
    
    t_current = 0
    a_current = 0
    
    for char in S:
        if char == 'T':
            t_current += 1
            if t_current == target_wins:
                print('T')
                break
        else:
            a_current += 1
            if a_current == target_wins:
                print('A')
                break