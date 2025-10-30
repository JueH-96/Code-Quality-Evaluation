# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
R = int(input[0])
G = int(input[1])
B = int(input[2])
C = input[3]

prices = {'Red': R, 'Green': G, 'Blue': B}
del prices[C]

print(min(prices.values()))