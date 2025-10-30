# YOUR CODE HERE

N = int(input())
S = input().strip()

t_total_wins = S.count('T')
a_total_wins = S.count('A')

if t_total_wins > a_total_wins:
    print('T')
elif a_total_wins > t_total_wins:
    print('A')
else:
    t_wins = 0
    a_wins = 0
    for c in S:
        if c == 'T':
            t_wins += 1
            if t_wins == t_total_wins:
                print('T')
                break
        else:
            a_wins += 1
            if a_wins == a_total_wins:
                print('A')
                break