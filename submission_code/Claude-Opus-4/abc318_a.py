# YOUR CODE HERE
N, M, P = map(int, input().split())

if M > N:
    print(0)
else:
    # Number of full moon days is floor((N-M)/P) + 1
    count = (N - M) // P + 1
    print(count)