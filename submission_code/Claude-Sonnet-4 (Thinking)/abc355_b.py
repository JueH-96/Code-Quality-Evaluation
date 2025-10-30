# Read input
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Create combined list with source information
combined = []
for val in a:
    combined.append((val, 'A'))
for val in b:
    combined.append((val, 'B'))

# Sort by value
combined.sort()

# Check for consecutive elements both from A
found = False
for i in range(len(combined) - 1):
    if combined[i][1] == 'A' and combined[i+1][1] == 'A':
        found = True
        break

# Output result
if found:
    print("Yes")
else:
    print("No")