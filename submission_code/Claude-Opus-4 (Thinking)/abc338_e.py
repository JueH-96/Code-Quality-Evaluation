n = int(input())
chords = []
for _ in range(n):
    a, b = map(int, input().split())
    chords.append((a, b))

def chords_intersect(chord1, chord2):
    a, b = chord1
    c, d = chord2
    
    # Ensure a < b and c < d for easier comparison
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
    
    # Check if the intervals (a, b) and (c, d) interleave
    # They interleave if one contains exactly one endpoint of the other
    return (a < c < b) != (a < d < b)

# Check all pairs of chords
found_intersection = False
for i in range(n):
    for j in range(i + 1, n):
        if chords_intersect(chords[i], chords[j]):
            found_intersection = True
            break
    if found_intersection:
        break

if found_intersection:
    print("Yes")
else:
    print("No")