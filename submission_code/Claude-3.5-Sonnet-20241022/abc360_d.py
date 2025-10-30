N, T = map(int, input().split())
S = input()
X = list(map(int, input().split()))

count = 0
for i in range(N):
    for j in range(i+1, N):
        # If ants are moving in opposite directions
        if S[i] != S[j]:
            # If ant i is moving right (1) and ant j is moving left (0)
            if S[i] == '1':
                if X[i] < X[j] and abs(X[j] - X[i])/2 <= T:
                    count += 1
            # If ant i is moving left (0) and ant j is moving right (1)
            else:
                if X[i] > X[j] and abs(X[i] - X[j])/2 <= T:
                    count += 1
        # If both ants are moving in same direction
        else:
            # Both moving right
            if S[i] == '1':
                if X[i] > X[j]:
                    count += 1
            # Both moving left
            else:
                if X[i] < X[j]:
                    count += 1

print(count)