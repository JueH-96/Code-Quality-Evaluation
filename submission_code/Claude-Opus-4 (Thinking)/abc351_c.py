# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

sequence = []

for i in range(n):
    sequence.append(a[i])
    
    while len(sequence) >= 2 and sequence[-1] == sequence[-2]:
        exp = sequence.pop()
        sequence.pop()
        sequence.append(exp + 1)

print(len(sequence))