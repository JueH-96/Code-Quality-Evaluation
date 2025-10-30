# YOUR CODE HERE
N_S_K = input().split()
N = int(N_S_K[0])
S = int(N_S_K[1])
K = int(N_S_K[2])

total = 0
for _ in range(N):
    P_Q = input().split()
    P = int(P_Q[0])
    Q = int(P_Q[1])
    total += P * Q

if total >= S:
    shipping_fee = 0
else:
    shipping_fee = K

amount = total + shipping_fee

print(amount)