import sys

input = sys.stdin.read
data = input().split()

A = int(data[0])
B = int(data[1])

# There are three suspects: 1, 2, and 3
suspects = {1, 2, 3}

# Remove the suspects that are not the culprit according to the witnesses
suspects.discard(A)
suspects.discard(B)

# If there is only one suspect left, that person is the culprit
if len(suspects) == 1:
    print(suspects.pop())
else:
    print(-1)