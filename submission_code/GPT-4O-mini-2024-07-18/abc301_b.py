def insert_numbers(sequence):
    while True:
        modified = False
        for i in range(len(sequence) - 1):
            if abs(sequence[i] - sequence[i + 1]) != 1:
                modified = True
                if sequence[i] < sequence[i + 1]:
                    # Insert numbers between sequence[i] and sequence[i + 1]
                    for num in range(sequence[i] + 1, sequence[i + 1]):
                        sequence.insert(i + 1, num)
                else:
                    # Insert numbers between sequence[i] and sequence[i + 1]
                    for num in range(sequence[i] - 1, sequence[i + 1], -1):
                        sequence.insert(i + 1, num)
                break
        if not modified:
            break
    return sequence

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N + 1]))

# Process the sequence
result = insert_numbers(A)

# Print the result
print(" ".join(map(str, result)))