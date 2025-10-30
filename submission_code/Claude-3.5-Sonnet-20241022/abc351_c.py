N = int(input())
A = list(map(int, input().split()))

sequence = []
for i in range(N):
    sequence.append(2**A[i])
    
    while len(sequence) >= 2:
        if sequence[-1] != sequence[-2]:
            break
        new_size = sequence.pop() + sequence.pop()
        sequence.append(new_size)

print(len(sequence))