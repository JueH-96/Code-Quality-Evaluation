# YOUR CODE HERE
def find_overall_winner(n, s):
    takahashi_wins = 0
    aoki_wins = 0
    
    for i in range(n):
        if s[i] == 'T':
            takahashi_wins += 1
        else:
            aoki_wins += 1
        
        # Check if one of them has already won more games than the other
        if takahashi_wins > aoki_wins:
            print('T')
            return
        elif aoki_wins > takahashi_wins:
            print('A')
            return
    
    # If we reach here, it means they have the same number of wins
    # The winner is the one who reached the number of wins first
    if takahashi_wins == aoki_wins:
        print('T' if s.rfind('T') > s.rfind('A') else 'A')

import sys
input = sys.stdin.read
data = input().split()
n = int(data[0])
s = data[1]

find_overall_winner(n, s)