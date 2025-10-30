N = int(input())
S = input()
T = input()

distance = 0
for i in range(N):
    if S[i] != T[i]:
        distance += 1

print(distance)