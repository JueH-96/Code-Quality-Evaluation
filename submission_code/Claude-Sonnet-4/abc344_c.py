# Read input
N = int(input())
A = list(map(int, input().split()))
M = int(input())
B = list(map(int, input().split()))
L = int(input())
C = list(map(int, input().split()))
Q = int(input())
X = list(map(int, input().split()))

# Precompute all possible sums from A and B
AB_sums = set()
for a in A:
    for b in B:
        AB_sums.add(a + b)

# Convert C to a set for O(1) lookup
C_set = set(C)

# Process each query
for x in X:
    found = False
    for ab_sum in AB_sums:
        needed_c = x - ab_sum
        if needed_c in C_set:
            found = True
            break
    
    if found:
        print("Yes")
    else:
        print("No")