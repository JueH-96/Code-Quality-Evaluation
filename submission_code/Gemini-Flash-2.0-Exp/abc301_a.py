N = int(input())
S = input()

takahashi_wins = 0
aoki_wins = 0

for char in S:
    if char == 'T':
        takahashi_wins += 1
    else:
        aoki_wins += 1

if takahashi_wins > aoki_wins:
    print('T')
elif aoki_wins > takahashi_wins:
    print('A')
else:
    takahashi_count = 0
    aoki_count = 0
    takahashi_first = False
    aoki_first = False
    
    for i in range(N):
        if S[i] == 'T':
            takahashi_count += 1
            if takahashi_count == takahashi_wins and not takahashi_first:
                takahashi_first = True
        else:
            aoki_count += 1
            if aoki_count == aoki_wins and not aoki_first:
                aoki_first = True
                
    if takahashi_first:
        print('T')
    else:
        print('A')