# Read input
N = int(input())
S = input()

# Find the position where "ABC" first appears
pos = S.find("ABC")
if pos != -1:
    print(pos + 1)  # Convert to 1-based indexing
else:
    print(-1)