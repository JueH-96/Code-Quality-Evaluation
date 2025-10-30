from collections import defaultdict

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# pair1 for the case where A[i] gets 1 and B[i] gets 0
# pair1 is the number of pairs (i, j) where i gets 1 and j gets 0.
pair1 = defaultdict(int)
# Initialize pair1 with 0 for each value from 1 to N
for i in range(1, N + 1):
    pair1[i] = 0

# pair2 for the case where A[i] gets 0 and B[i] gets 1
pair2 = defaultdict(int)
# Initialize pair2 with 0 for each value from 1 to N
for i in range(1, N + 1):
    pair2[i] = 0

# Increment pair1 or pair2 based on whether A[i] or B[i] is assigned 1
for i in range(M):
    pair1[A[i]] += 1
    pair2[B[i]] += 1

# Increment pair1 or pair2 based on whether A[i] or B[i] is assigned 0
for i in range(M):
    pair1[B[i]] += 1
    pair2[A[i]] += 1

# Initialize the count of ranges where X can be given any value to 0
X = 0
# Check if there exists a value of X for which both pair1 and pair2 values are greater than 0
for i in range(1, N + 1):
    if pair1[i] > 0 and pair2[i] > 0:
        X = 1  # If such a value is found, set X to 1 and break the loop
        break

if X:
    print("Yes")
else:
    print("No")