cards = list(map(int, input().split()))
from collections import defaultdict

freq = defaultdict(int)
for num in cards:
    freq[num] += 1

# Check all possible pairs of x and y (x != y) where x's count >=3 and y's count >=2
found = False
keys = list(freq.keys())
for x in keys:
    if freq[x] >= 3:
        remaining = [y for y in keys if y != x]
        for y in remaining:
            if freq[y] >= 2:
                found = True
                break
        if found:
            break
    if found:
        break

if not found:
    # Check if there's a number with >=5 counts (but then no full house possible as per note)
    # But sample input 3 shows that 5 identical cards is not a full house
    pass

print("Yes" if found else "No")