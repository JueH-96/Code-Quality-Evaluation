import sys

input = sys.stdin.readline

for _ in range(int(input())):
    S = input().strip()
    X = input().strip()
    Y = input().strip()

    if len(S) == 0:
        print("Yes")
        continue

    if len(X) != len(Y):
        print("No")
        continue

    start = S[0]
    end = S[-1]

    xCounts = [0] * len(X)
    yCounts = [0] * len(Y)
    xCounts[0] = 1 if int(X[0]) == 0 else 0
    yCounts[0] = 1 if int(Y[0]) == 0 else 0
    for i in range(1, len(X)):
        if X[i] == "0":
            xCounts[i] = xCounts[i - 1] + 1
        else:
            xCounts[i] = xCounts[i - 1]

        if Y[i] == "0":
            yCounts[i] = yCounts[i - 1] + 1
        else:
            yCounts[i] = yCounts[i - 1]

    if any(
        True for i in range(len(X))
        if X[i] != Y[i]
        and (start != end or xCounts[i] % len(S) != yCounts[i] % len(S))
    ):
        print("No")
    else:
        print("Yes")