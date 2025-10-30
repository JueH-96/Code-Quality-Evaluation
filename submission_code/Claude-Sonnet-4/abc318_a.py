N, M, P = map(int, input().split())

if M > N:
    print(0)
else:
    # Number of full moons = number of valid k values where M + k*P <= N
    # k can be 0, 1, 2, ..., floor((N-M)/P)
    max_k = (N - M) // P
    print(max_k + 1)