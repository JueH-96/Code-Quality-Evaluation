# YOUR CODE HERE

from sys import stdin

N = int(stdin.readline())
A = list(map(int, stdin.readline().split()))

sequence = []

for i in range(N):
    sequence.append(2**A[i])
    while len(sequence) > 1 and sequence[-1] == sequence[-2]:
        sequence.pop()
        sequence.pop()
        sequence.append(2**(A[i]+1))

print(len(sequence))