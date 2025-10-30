import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S = int(data[index])
index += 1
K = int(data[index])
index += 1
total_price = 0
for i in range(N):
    P = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    total_price += P * Q
if total_price >= S:
    shipping_fee = 0
else:
    shipping_fee = K
total_amount = total_price + shipping_fee
print(total_amount)