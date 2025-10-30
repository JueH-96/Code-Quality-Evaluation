import sys

# Read input
N, M, H, K = map(int, input().split())
S = input()
positions = []
for _ in range(M):
    x, y = map(int, input().split())
    positions.append((x, y))

# Simulate Takahashi's moves
x, y = 0, 0
health = H
for i in range(N):
    if health <= 0:
        print("No")
        return
    
    if S[i] == "R":
        x += 1
    elif S[i] == "L":
        x -= 1
    elif S[i] == "U":
        y += 1
    else:
        y -= 1
    
    health -= 1
    
    if (x, y) in positions and health < K:
        health = K

print("Yes")