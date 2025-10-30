N, M, P = map(int, input().split())

if M > N:
    print(0)
else:
    print((N - M) // P + 1)