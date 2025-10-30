# Read input
N, M, P = map(int, input().split())

# Check if M is beyond N
if M > N:
    print(0)
else:
    # Calculate the number of terms in the arithmetic sequence
    count = (N - M) // P + 1
    print(count)