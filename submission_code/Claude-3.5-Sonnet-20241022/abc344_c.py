N = int(input())
A = list(map(int, input().split()))
M = int(input())
B = list(map(int, input().split()))
L = int(input())
C = list(map(int, input().split()))
Q = int(input())
X = list(map(int, input().split()))

# Create set of all possible sums of A and B
ab_sums = set()
for a in A:
    for b in B:
        ab_sums.add(a + b)

# For each query
for x in X:
    possible = False
    # Check if x - c equals any sum of a + b
    for c in C:
        if x - c in ab_sums:
            possible = True
            break
    print("Yes" if possible else "No")