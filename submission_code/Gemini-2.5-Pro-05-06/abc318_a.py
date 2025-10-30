# Read the input values for N, M, and P
N, M, P = map(int, input().split())

# If the first full moon day M is greater than N,
# then no full moons occur in the range [1, N].
if M > N:
    ans = 0
else:
    # Otherwise, M is within the range [1, N].
    # The full moon days are M, M+P, M+2P, ..., M+kP.
    # We need to find the number of terms such that M + kP <= N.
    # This implies kP <= N - M.
    # Since P > 0, k <= (N - M) / P.
    # The values of k are 0, 1, ..., floor((N - M) / P).
    # The number of such values is floor((N - M) / P) + 1.
    # In Python, floor division is done by //.
    ans = (N - M) // P + 1

# Print the calculated answer
print(ans)