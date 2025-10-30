# Read inputs
N = int(input())
A = list(map(int, input().split()))
M = int(input())
B = list(map(int, input().split()))
L = int(input())
C = list(map(int, input().split()))
Q = int(input())
X = list(map(int, input().split()))

# Generate all possible sums
possible_sums = set()
for a in A:
    for b in B:
        for c in C:
            possible_sums.add(a + b + c)

# Answer queries
for x in X:
    if x in possible_sums:
        print("Yes")
    else:
        print("No")