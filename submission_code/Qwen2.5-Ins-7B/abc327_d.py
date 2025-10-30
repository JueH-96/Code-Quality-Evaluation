# N M
# A_1 A_2 ... A_M
# B_1 B_2 ... B_M
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Create a list to store the difference between A and B for each index
diff = [a - b for a, b in zip(A, B)]

# Create a set to store the unique differences
unique_diff = set(diff)

# Check if the set of unique differences contains both 0 and a non-zero value
if 0 in unique_diff and any(d != 0 for d in unique_diff):
    print("Yes")
else:
    print("No")