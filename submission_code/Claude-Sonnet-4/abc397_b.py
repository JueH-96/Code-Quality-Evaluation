# YOUR CODE HERE
s = input().strip()
insertions = 0
pos = 0  # position in the target string (0-indexed)

for char in s:
    # At position pos, we expect 'i' if pos is even, 'o' if pos is odd
    expected = 'i' if pos % 2 == 0 else 'o'
    
    while char != expected:
        # Insert the expected character
        insertions += 1
        pos += 1
        expected = 'i' if pos % 2 == 0 else 'o'
    
    # Now char matches expected, so we place it
    pos += 1

# If final length is odd, add one more character
if pos % 2 == 1:
    insertions += 1

print(insertions)