N = int(input())
S = input()

t_wins = 0
a_wins = 0
t_final = 0
a_final = 0

for i in range(N):
    if S[i] == 'T':
        t_wins += 1
        if t_wins > a_wins and t_final == 0:
            t_final = i + 1
    else:
        a_wins += 1
        if a_wins > t_wins and a_final == 0:
            a_final = i + 1

if t_wins > a_wins:
    print('T')
elif a_wins > t_wins:
    print('A')
else:
    if t_final < a_final:
        print('T')
    else:
        print('A')