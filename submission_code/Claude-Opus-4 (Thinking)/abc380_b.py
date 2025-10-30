# YOUR CODE HERE
S = input().strip()

# Split by "|", filter out empty strings, and count lengths
A = [len(s) for s in S.split('|') if s]

print(' '.join(map(str, A)))