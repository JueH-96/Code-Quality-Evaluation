# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

P = []
for i in range(N):
    P.append(data[index])
    index += 1

queries = []
for i in range(Q):
    queries.append([int(data[index]), int(data[index + 1]), int(data[index + 2]), int(data[index + 3])])
    index += 4

def count_black_squares(A, B, C, D):
    count = 0
    for i in range(A, C + 1):
        for j in range(B, D + 1):
            if P[i % N][j % N] == 'B':
                count += 1
    return count

for query in queries:
    A, B, C, D = query
    print(count_black_squares(A, B, C, D))