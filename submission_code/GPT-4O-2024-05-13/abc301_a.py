# YOUR CODE HERE
def find_winner(N, S):
    takahashi_wins = 0
    aoki_wins = 0
    
    for i in range(N):
        if S[i] == 'T':
            takahashi_wins += 1
        else:
            aoki_wins += 1
        
        if takahashi_wins > aoki_wins:
            print('T')
            return
        elif aoki_wins > takahashi_wins:
            print('A')
            return
    
    # If the loop completes, it means they have the same number of wins
    if takahashi_wins == aoki_wins:
        print('T' if S.rfind('T') > S.rfind('A') else 'A')

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

find_winner(N, S)