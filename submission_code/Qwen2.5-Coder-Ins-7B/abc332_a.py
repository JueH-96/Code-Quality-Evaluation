# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = int(data[1])
K = int(data[2])

total_price = 0
for i in range(N):
    P = int(data[3 + 2 * i])
    Q = int(data[4 + 2 * i])
    total_price += P * Q

if total_price >= S:
    shipping_fee = 0
else:
    shipping_fee = K

print(total_price + shipping_fee)