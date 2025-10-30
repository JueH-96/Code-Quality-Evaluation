from collections import Counter

# Read input
A, B, C, D = map(int, input().split())

found = False

# Check each possible card from 1 to 13
for x in range(1, 14):
    temp = [A, B, C, D, x]
    cnt = Counter(temp)
    # Check if there are exactly two distinct values and their counts are 3 and 2
    if len(cnt) == 2 and sorted(cnt.values()) == [2, 3]:
        found = True
        break

print("Yes" if found else "No")