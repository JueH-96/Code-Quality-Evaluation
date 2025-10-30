# Read input
N, M, P = map(int, input().split())

# Check if the first day is beyond N
if M > N:
    print(0)
else:
    # Calculate the number of terms in the arithmetic sequence
    diff = N - M
    count = (diff // P) + 1
    print(count)