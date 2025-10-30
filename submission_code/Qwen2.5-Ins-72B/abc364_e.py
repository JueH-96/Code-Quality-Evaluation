# YOUR CODE HERE
import sys
from itertools import permutations

input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])
A = [int(data[3 + 2 * i]) for i in range(N)]
B = [int(data[4 + 2 * i]) for i in range(N)]

def max_dishes(A, B, X, Y):
    max_eaten = 0
    for perm in permutations(range(N)):
        sweetness = 0
        saltiness = 0
        eaten = 0
        for i in perm:
            if sweetness + A[i] > X or saltiness + B[i] > Y:
                break
            sweetness += A[i]
            saltiness += B[i]
            eaten += 1
        max_eaten = max(max_eaten, eaten)
    return max_eaten

print(max_dishes(A, B, X, Y))