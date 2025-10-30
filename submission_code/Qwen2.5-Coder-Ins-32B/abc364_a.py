# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
dishes = input[1:]

for i in range(1, N):
    if dishes[i] == "sweet" and dishes[i-1] == "sweet":
        print("No")
        break
else:
    print("Yes")