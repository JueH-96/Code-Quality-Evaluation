# YOUR CODE HERE
N = int(input())
S = input()

t_wins = 0
a_wins = 0

for i in range(N):
    if S[i] == 'T':
        t_wins += 1
    else:
        a_wins += 1
    
    if t_wins > a_wins:
        print('T')
        break
    elif a_wins > t_wins:
        print('A')
        break
else:
    if t_wins == a_wins:
        if S[-1] == 'T':
            print('T')
        else:
            print('A')
    elif t_wins > a_wins:
        print('T')
    else:
        print('A')