# Read input
Q = int(input())
sequence = []
output = []

for _ in range(Q):
    parts = input().split()
    if parts[0] == '1':
        # Append operation
        x = int(parts[1])
        sequence.append(x)
    else:
        # Query operation
        k = int(parts[1])
        output.append(str(sequence[-k]))

# Print all results
print('
'.join(output))