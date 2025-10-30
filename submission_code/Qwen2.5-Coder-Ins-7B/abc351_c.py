# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

sequence = []

for a in A:
    sequence.append(2 ** a)
    
    while len(sequence) > 1 and sequence[-1] == sequence[-2]:
        sequence.pop()
        sequence.pop()
        if sequence:
            sequence.append(sequence[-1] + sequence[-1])
    
print(len(sequence))