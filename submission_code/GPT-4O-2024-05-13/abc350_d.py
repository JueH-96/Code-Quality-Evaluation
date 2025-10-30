# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

friendships = defaultdict(set)

index = 2
for _ in range(M):
    A = int(data[index])
    B = int(data[index + 1])
    friendships[A].add(B)
    friendships[B].add(A)
    index += 2

count = 0

for Y in range(1, N + 1):
    friends_of_Y = list(friendships[Y])
    for i in range(len(friends_of_Y)):
        for j in range(i + 1, len(friends_of_Y)):
            X = friends_of_Y[i]
            Z = friends_of_Y[j]
            if Z not in friendships[X]:
                count += 1
                friendships[X].add(Z)
                friendships[Z].add(X)

print(count)