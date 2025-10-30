def can_form_sum(a, b, c, target):
    # Pre-compute all possible sums of elements from a and b
    ab_sums = set()
    for ai in a:
        for bi in b:
            ab_sums.add(ai + bi)
    
    # Check if we can form the target sum
    for ci in c:
        if target - ci in ab_sums:
            return True
    
    return False

# Read input
N = int(input())
A = list(map(int, input().split()))
M = int(input())
B = list(map(int, input().split()))
L = int(input())
C = list(map(int, input().split()))
Q = int(input())
X = list(map(int, input().split()))

# Pre-compute all possible sums of elements from A and B
AB_sums = set()
for a in A:
    for b in B:
        AB_sums.add(a + b)

# Process each query
for x in X:
    found = False
    for c in C:
        if x - c in AB_sums:
            found = True
            break
    
    if found:
        print("Yes")
    else:
        print("No")