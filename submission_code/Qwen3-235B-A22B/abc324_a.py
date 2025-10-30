# Read input values
n = int(input())
a = list(map(int, input().split()))

# Check if all elements are the same using a set
if len(set(a)) == 1:
    print("Yes")
else:
    print("No")