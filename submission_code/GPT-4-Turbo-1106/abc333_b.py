# Read the input from stdin
S = input().strip()
T = input().strip()

# Define the pentagon points in order
pentagon = 'ABCDE'

# Calculate the distance between points on the pentagon
def distance(p1, p2):
    idx1, idx2 = pentagon.index(p1), pentagon.index(p2)
    return min(abs(idx1 - idx2), 5 - abs(idx1 - idx2))

# Check if the distances are equal
if distance(S[0], S[1]) == distance(T[0], T[1]):
    print("Yes")
else:
    print("No")