import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = int(data[1])
K = int(data[2])

index = 3
total_price = 0

for i in range(N):
    P = int(data[index])
    Q = int(data[index + 1])
    total_price += P * Q
    index += 2

if total_price >= S:
    shipping_fee = 0
else:
    shipping_fee = K

total_amount = total_price + shipping_fee
print(total_amount)