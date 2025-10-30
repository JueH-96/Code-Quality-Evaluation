# YOUR CODE HERE
N = int(input())
S = input()

takahashi_wins = S.count('T')
aoki_wins = S.count('A')

if takahashi_wins > aoki_wins:
    print('T')
elif aoki_wins > takahashi_wins:
    print('A')
else:
    # They have the same number of wins
    # Find who reached that number first
    target_wins = takahashi_wins
    
    takahashi_count = 0
    aoki_count = 0
    
    for i in range(N):
        if S[i] == 'T':
            takahashi_count += 1
            if takahashi_count == target_wins:
                print('T')
                break
        else:
            aoki_count += 1
            if aoki_count == target_wins:
                print('A')
                break