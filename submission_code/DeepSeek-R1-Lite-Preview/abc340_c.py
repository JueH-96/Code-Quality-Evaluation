def total_payment(N):
    if N == 1:
        return 0
    k = N.bit_length() - 1
    m = N - (1 << k)
    return N * k + 2 * m

# Read input
N = int(input())

# Compute and print the total payment
print(total_payment(N))