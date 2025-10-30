# Read input
n = int(input())
a = list(map(int, input().split()))
m = int(input())
b = list(map(int, input().split()))
l = int(input())
c = list(map(int, input().split()))
q = int(input())
xs = list(map(int, input().split()))

# Compute all possible a + b sums
sum_ab = set()
for ai in a:
    for bi in b:
        sum_ab.add(ai + bi)

# Compute all possible a + b + c sums
possible_sums = set()
for s in sum_ab:
    for ci in c:
        possible_sums.add(s + ci)

# Check each query and output result
for x in xs:
    print("Yes" if x in possible_sums else "No")