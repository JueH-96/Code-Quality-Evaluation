# YOUR CODE HERE
N = int(input())
S = input().strip()

takahashi_wins = 0
aoki_wins = 0

for i, result in enumerate(S):
    if result == 'T':
        takahashi_wins += 1
    else:
        aoki_wins += 1
    
    if takahashi_wins > N // 2 or aoki_wins > N // 2:
        break

if takahashi_wins > aoki_wins:
    print('T')
elif aoki_wins > takahashi_wins:
    print('A')
else:
    print('T' if S.index('T') < S.index('A') else 'A')