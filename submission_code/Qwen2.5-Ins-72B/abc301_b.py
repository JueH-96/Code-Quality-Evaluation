# YOUR CODE HERE
def insert_numbers(sequence):
    while True:
        inserted = False
        new_sequence = []
        for i in range(len(sequence) - 1):
            new_sequence.append(sequence[i])
            diff = abs(sequence[i] - sequence[i + 1])
            if diff > 1:
                if sequence[i] < sequence[i + 1]:
                    new_sequence.extend(range(sequence[i] + 1, sequence[i + 1]))
                else:
                    new_sequence.extend(range(sequence[i] - 1, sequence[i + 1], -1))
                inserted = True
        new_sequence.append(sequence[-1])
        if not inserted:
            break
        sequence = new_sequence
    return sequence

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

result = insert_numbers(A)
print(' '.join(map(str, result)))