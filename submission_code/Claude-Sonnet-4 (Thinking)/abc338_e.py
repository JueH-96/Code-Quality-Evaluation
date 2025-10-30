def chords_intersect(a, b, c, d):
    """Check if chord (a,b) intersects with chord (c,d)"""
    # Sort all four endpoints with their chord labels
    points = [(a, 1), (b, 1), (c, 2), (d, 2)]
    points.sort()
    
    # Check if the pattern is alternating
    pattern = [point[1] for point in points]
    return pattern == [1, 2, 1, 2] or pattern == [2, 1, 2, 1]

n = int(input())
chords = []
for _ in range(n):
    a, b = map(int, input().split())
    chords.append((a, b))

# Check all pairs of chords
intersect = False
for i in range(n):
    for j in range(i + 1, n):
        if chords_intersect(chords[i][0], chords[i][1], chords[j][0], chords[j][1]):
            intersect = True
            break
    if intersect:
        break

print("Yes" if intersect else "No")