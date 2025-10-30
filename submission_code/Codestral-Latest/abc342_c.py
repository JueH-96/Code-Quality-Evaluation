# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = list(data[1])
Q = int(data[2])

operations = data[3:3 + 2 * Q]
operations = [operations[i:i + 2] for i in range(0, len(operations), 2)]

# Create a dictionary to store the final character for each character
final_char = {chr(i): chr(i) for i in range(ord('a'), ord('z') + 1)}

# Process the operations in reverse order to determine the final character for each character
for c, d in reversed(operations):
    final_char[c] = final_char[d]

# Replace each character in the string S with its final character
result = ''.join(final_char[char] for char in S)

print(result)