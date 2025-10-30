N, M = map(int, input().split())

if N == 1:
    X = M + 1
else:
    # Using the geometric series formula
    X = (pow(N, M + 1) - 1) // (N - 1)

if X > 10**9:
    print("inf")
else:
    print(X)