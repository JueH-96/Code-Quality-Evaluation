from collections import Counter

# Read the input
a = list(map(int, input().split()))
counts = Counter(a)

found = False

# Check each key in the counts
for key in counts:
    if counts[key] >= 3:
        # Check if there's another key with count >=2
        for other in counts:
            if other != key and counts[other] >= 2:
                found = True
                break
        if found:
            break

print("Yes" if found else "No")