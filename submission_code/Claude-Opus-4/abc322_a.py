# YOUR CODE HERE
N = int(input())
S = input()

# Find the first occurrence of "ABC" in S
position = S.find("ABC")

# If found, convert from 0-based to 1-based indexing
if position != -1:
    print(position + 1)
else:
    print(-1)