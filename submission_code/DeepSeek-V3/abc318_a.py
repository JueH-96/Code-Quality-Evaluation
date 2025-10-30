# YOUR CODE HERE
N, M, P = map(int, input().split())

if M > N:
    print(0)
else:
    # Calculate the number of terms in the arithmetic sequence
    # The sequence is M, M+P, M+2P, ..., M+kP <= N
    # So, k = floor((N - M) / P)
    k = (N - M) // P
    # The total number of days is k + 1 (since we include M itself)
    print(k + 1)