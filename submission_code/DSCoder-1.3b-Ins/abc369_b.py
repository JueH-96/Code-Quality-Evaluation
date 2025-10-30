# YOUR CODE HERE

N = int(input())
keys = []
for _ in range(N):
    A, S = map(int, input().split())
    keys.append((A, S))

keys.sort()

fatigue = 0
for A, S in keys:
    fatigue += abs(A - S)

print(fatigue)