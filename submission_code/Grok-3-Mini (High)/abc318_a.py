# Read input from stdin
N, M, P = map(int, input().split())

# Check if M > N, if so, no full moon days
if M > N:
    print(0)
else:
    # Calculate the number of full moon days using the formula
    ans = (N - M) // P + 1
    print(ans)