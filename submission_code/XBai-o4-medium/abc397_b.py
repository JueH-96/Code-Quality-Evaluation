S = input().strip()
expected = 'i'
insertions = 0

for c in S:
    while c != expected:
        insertions += 1
        expected = 'o' if expected == 'i' else 'i'
    # Toggle the expected after processing current character
    expected = 'o' if expected == 'i' else 'i'

# Check if the final length is even
if expected != 'i':
    insertions += 1

print(insertions)