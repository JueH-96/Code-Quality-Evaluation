# Read N, S, K
N_S_K = input().split()
N = int(N_S_K[0])
S = int(N_S_K[1])
K = int(N_S_K[2])

# Read N lines of P_i and Q_i
total_price = 0
for _ in range(N):
    P_Q = input().split()
    P = int(P_Q[0])
    Q = int(P_Q[1])
    total_price += P * Q

# Determine shipping_fee
if total_price >= S:
    shipping_fee = 0
else:
    shipping_fee = K

# Calculate total_payment
total_payment = total_price + shipping_fee

# Print the result
print(total_payment)