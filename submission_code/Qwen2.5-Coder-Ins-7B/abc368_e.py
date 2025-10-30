import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
X1 = int(data[2])

trains = []
for i in range(M):
    A = int(data[3 + 4 * i])
    B = int(data[4 + 4 * i])
    S = int(data[5 + 4 * i])
    T = int(data[6 + 4 * i])
    trains.append((A, B, S, T))

# Initialize X array with X1
X = [X1] + [0] * (M - 1)

# Process each train to find the minimum delay
for i in range(M):
    A, B, S, T = trains[i]
    for j in range(M):
        if i != j:
            A2, B2, S2, T2 = trains[j]
            if B == A2 and T <= S2:
                X[j] = max(X[j], T + X[i] - S2)

# Print the result
print(" ".join(map(str, X[1:])))