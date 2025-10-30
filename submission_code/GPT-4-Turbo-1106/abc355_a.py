# Read the inputs from stdin
A, B = map(int, input().split())

# Determine the culprit based on the memories of the two witnesses
if A != B:
    # If A and B are different, the culprit is the one not mentioned by either witness
    culprit = 6 - A - B
else:
    # If A and B are the same, the culprit cannot be uniquely identified
    culprit = -1

# Write the answer to stdout
print(culprit)