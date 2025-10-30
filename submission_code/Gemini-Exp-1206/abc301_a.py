n = int(input())
s = input()
takahashi_wins = 0
aoki_wins = 0
for char in s:
    if char == 'T':
        takahashi_wins += 1
    else:
        aoki_wins += 1
if takahashi_wins > aoki_wins:
    print('T')
elif aoki_wins > takahashi_wins:
    print('A')
else:
    takahashi_wins = 0
    aoki_wins = 0
    for char in s:
        if char == 'T':
            takahashi_wins += 1
        else:
            aoki_wins += 1
        if takahashi_wins > aoki_wins and takahashi_wins >= (n + 1) // 2 :
            print('T')
            break
        elif aoki_wins > takahashi_wins and aoki_wins >= (n + 1) // 2:
            print('A')
            break
        elif takahashi_wins == aoki_wins and takahashi_wins >= (n+1)//2:
            if char == 'T':
                print('T')
            else:
                print('A')
            break