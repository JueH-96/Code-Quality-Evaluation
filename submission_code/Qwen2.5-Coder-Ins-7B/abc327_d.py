# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+M]))
B = list(map(int, data[2+M:]))

# Check if there exists a sequence X of length N consisting of 0 and 1
# such that X_{A_i} != X_{B_i} for each i=1, 2, ..., M
def is_good_pair(A, B):
    # Create a set to store the pairs (A_i, B_i)
    pairs = set()
    for a, b in zip(A, B):
        if (a, b) in pairs or (b, a) in pairs:
            return False
        pairs.add((a, b))
    return True

if is_good_pair(A, B):
    print("Yes")
else:
    print("No")