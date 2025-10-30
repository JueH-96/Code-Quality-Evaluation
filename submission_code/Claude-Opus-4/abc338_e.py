# YOUR CODE HERE
n = int(input())
chords = []
for _ in range(n):
    a, b = map(int, input().split())
    # Ensure a < b for easier comparison
    if a > b:
        a, b = b, a
    chords.append((a, b))

# Check all pairs of chords for intersection
intersect = False
for i in range(n):
    for j in range(i + 1, n):
        a1, b1 = chords[i]
        a2, b2 = chords[j]
        
        # Two chords intersect if their endpoints alternate
        # i.e., a1 < a2 < b1 < b2 or a2 < a1 < b2 < b1
        if (a1 < a2 < b1 < b2) or (a2 < a1 < b2 < b1):
            intersect = True
            break
    if intersect:
        break

if intersect:
    print("Yes")
else:
    print("No")