# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
S = int(input[1])
K = int(input[2])

total_price = 0
index = 3
for _ in range(N):
    P_i = int(input[index])
    Q_i = int(input[index + 1])
    total_price += P_i * Q_i
    index += 2

if total_price < S:
    total_price += K

print(total_price)