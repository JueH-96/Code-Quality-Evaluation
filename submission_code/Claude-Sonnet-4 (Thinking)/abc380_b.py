S = input().strip()

# Remove the first and last "|" and split by "|"
parts = S[1:-1].split("|")

# For each part, count the number of "-" characters
A = [len(part) for part in parts]

print(" ".join(map(str, A)))