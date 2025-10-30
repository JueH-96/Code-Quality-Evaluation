# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Create combined list with source information
# Each element is (value, source) where source is 'A' or 'B'
combined = []
for val in A:
    combined.append((val, 'A'))
for val in B:
    combined.append((val, 'B'))

# Sort by value
combined.sort()

# Check for consecutive elements both from A
found = False
for i in range(len(combined) - 1):
    if combined[i][1] == 'A' and combined[i+1][1] == 'A':
        found = True
        break

if found:
    print("Yes")
else:
    print("No")