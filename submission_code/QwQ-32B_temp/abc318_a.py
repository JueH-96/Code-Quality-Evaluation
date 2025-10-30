# Read input values
N, M, P = map(int, input().split())

# Check if the first possible day is after N
if M > N:
    print(0)
else:
    # Calculate the number of terms in the arithmetic sequence
    count = (N - M) // P + 1
    print(count)