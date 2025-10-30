# YOUR CODE HERE
n = int(input())
s = input()
takahashi_wins = 0
aoki_wins = 0
takahashi_first_win = -1
aoki_first_win = -1
for i in range(n):
    if s[i] == 'T':
        takahashi_wins += 1
        if takahashi_first_win == -1:
            takahashi_first_win = i
    else:
        aoki_wins += 1
        if aoki_first_win == -1:
            aoki_first_win = i
if takahashi_wins > aoki_wins:
    print('T')
elif aoki_wins > takahashi_wins:
    print('A')
else:
    if takahashi_first_win < aoki_first_win:
        print('T')
    else:
        print('A')