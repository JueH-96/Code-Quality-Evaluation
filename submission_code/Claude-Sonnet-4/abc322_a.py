# YOUR CODE HERE
N = int(input())
S = input().strip()

position = S.find("ABC")

if position == -1:
    print(-1)
else:
    print(position + 1)  # Convert from 0-based to 1-based indexing