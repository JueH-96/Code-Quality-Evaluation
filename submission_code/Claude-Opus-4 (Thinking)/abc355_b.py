# YOUR CODE HERE
# Read input
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Create C by sorting all elements from A and B
c = sorted(a + b)

# Create a set of elements in A for quick lookup
a_set = set(a)

# Check if there are two consecutive elements in C that both came from A
found = False
for i in range(len(c) - 1):
    if c[i] in a_set and c[i+1] in a_set:
        found = True
        break

if found:
    print("Yes")
else:
    print("No")