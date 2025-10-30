# YOUR CODE HERE
N = int(input())
S = input()

# Find the first occurrence of "ABC"
pos = S.find("ABC")

if pos == -1:
    print(-1)
else:
    # Convert to 1-indexed position
    print(pos + 1)