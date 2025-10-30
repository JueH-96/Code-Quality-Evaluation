# YOUR CODE HERE
def game_winner(N, A):
    # Determine the Grundy number for each A_i
    def grundy(x):
        if x == 1:
            return 0
        if x == 2:
            return 1
        if x == 3:
            return 1
        if x % 2 == 0:
            return x // 2
        return 0
    
    # Calculate the nim-sum (XOR of all Grundy numbers)
    nim_sum = 0
    for a in A:
        nim_sum ^= grundy(a)
    
    # If nim-sum is 0, Bruno wins; otherwise, Anna wins
    if nim_sum == 0:
        return "Bruno"
    else:
        return "Anna"

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Determine the winner and print the result
print(game_winner(N, A))