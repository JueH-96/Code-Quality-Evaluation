# Read input
n = int(input())
a = list(map(int, input().split()))
m = int(input())
b = list(map(int, input().split()))
l = int(input())
c = list(map(int, input().split()))
q = int(input())
x = list(map(int, input().split()))

# Precompute all possible sums of a + b + c
sum_set = set()
for ai in a:
    for bi in b:
        for ci in c:
            sum_set.add(ai + bi + ci)

# Process each query
for xi in x:
    print("Yes" if xi in sum_set else "No")