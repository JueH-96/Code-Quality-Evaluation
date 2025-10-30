a, b, c, d, e = map(int, input().split())
score_map = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}

participants = []
for i in range(1, 32):
    # Convert i to a 5-bit binary string
    binary = bin(i)[2:].zfill(5)
    # Generate the name based on the binary representation
    name = ''.join(['ABCDE'[j] for j in range(5) if binary[j] == '1'])
    # Calculate the score
    score = sum(score_map[char] for char in name)
    # Append as a tuple for sorting
    participants.append((-score, name))

# Sort the participants by score (descending) and lex order (ascending)
participants.sort()

# Extract the names in the sorted order
result = [name for (neg_score, name) in participants]

# Print each name
for name in result:
    print(name)