# Read the input string S from standard input
S = input().strip()

# Split S by "|" and ignore the first and last empty strings
segments = S.split('|')[1:-1]

# For each segment, count the number of "-" characters
A = [len(segment) for segment in segments]

# Print the sequence A as space-separated integers
print(' '.join(map(str, A)))