# YOUR CODE HERE
# Read input
N = int(input())
A = list(map(int, input().split()))

M = int(input())
B = list(map(int, input().split()))

L = int(input())
C = list(map(int, input().split()))

Q = int(input())
X = list(map(int, input().split()))

# Precompute all possible sums of elements from A and B
ab_sums = set()
for a in A:
    for b in B:
        ab_sums.add(a + b)

# For each query
for x in X:
    found = False
    for c in C:
        if (x - c) in ab_sums:
            found = True
            break
    
    if found:
        print("Yes")
    else:
        print("No")