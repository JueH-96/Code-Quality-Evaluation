import math

# Read the input
N = int(input())
dice = []
for _ in range(N):
    K = list(map(int, input().split()))
    K_i = K[0]
    A_i = K[1:]
    dice.append((K_i, A_i))

# Calculate the maximum probability
max_prob = 0
for i in range(N):
    for j in range(i+1, N):
        K_i, A_i = dice[i]
        K_j, A_j = dice[j]
        prob = 0
        for a in A_i:
            for b in A_j:
                if a == b:
                    prob += 1 / (K_i * K_j)
        max_prob = max(max_prob, prob)

print(f"{max_prob:.15f}")