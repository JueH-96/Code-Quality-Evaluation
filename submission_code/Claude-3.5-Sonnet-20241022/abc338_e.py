def has_intersection(a, b, c, d):
    # Check if segments (a,b) and (c,d) intersect
    # Convert to clockwise order
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
        
    # Check if one segment is completely inside the other
    if (a < c < b and a < d < b) or (c < a < d and c < b < d):
        return False
        
    # Check if segments intersect
    return (a < c < b and b < d) or (c < a < d and d < b)

N = int(input())
chords = []
for _ in range(N):
    a, b = map(int, input().split())
    chords.append((a, b))

# Check all pairs of chords
has_intersect = False
for i in range(N):
    for j in range(i+1, N):
        if has_intersection(chords[i][0], chords[i][1], chords[j][0], chords[j][1]):
            has_intersect = True
            break
    if has_intersect:
        break

print("Yes" if has_intersect else "No")