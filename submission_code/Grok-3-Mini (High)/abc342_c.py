import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Read the string S
S_str = data[index]
index += 1

# Read Q
Q = int(data[index])
index += 1

# Initialize the mapping for all 26 lowercase English letters
mapping = [chr(i + ord('a')) for i in range(26)]

# Perform Q operations
for _ in range(Q):
    c_i = data[index]  # Character to replace
    index += 1
    d_i = data[index]  # Character to replace with
    index += 1
    
    # Find all base indices where the current mapping equals c_i and update to d_i
    for x in range(26):
        if mapping[x] == c_i:
            mapping[x] = d_i

# Build the result string by applying the mapping to each character in S_str
result = []
for char in S_str:
    idx = ord(char) - ord('a')  # Get the index of the character
    mapped_char = mapping[idx]  # Get the mapped character
    result.append(mapped_char)

# Output the result string
print(''.join(result))