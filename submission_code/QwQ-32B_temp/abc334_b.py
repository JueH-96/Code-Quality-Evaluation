A, M, L, R = map(int, input().split())

numerator_low = L - A
numerator_high = R - A

# Compute lower_k as ceil(numerator_low / M)
lower_k = (numerator_low + M - 1) // M
# Compute upper_k as floor(numerator_high / M)
upper_k = numerator_high // M

if upper_k < lower_k:
    print(0)
else:
    print(upper_k - lower_k + 1)