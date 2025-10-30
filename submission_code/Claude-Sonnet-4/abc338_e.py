def intersect(a, b, c, d, n):
    # Check if chord (a,b) intersects with chord (c,d)
    # Normalize so that a < b and c < d for easier comparison
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
    
    # Two chords intersect if their endpoints alternate around the circle
    # This happens when exactly one endpoint of each chord lies in the arc
    # defined by the other chord
    
    def in_arc(x, p, q):
        # Check if x is in the clockwise arc from p to q
        if p < q:
            return p < x < q
        else:  # p > q, arc crosses the "wrap point"
            return x > p or x < q
    
    # Check if exactly one of c,d is in arc from a to b
    c_in_ab = in_arc(c, a, b)
    d_in_ab = in_arc(d, a, b)
    
    # Check if exactly one of a,b is in arc from c to d  
    a_in_cd = in_arc(a, c, d)
    b_in_cd = in_arc(b, c, d)
    
    # Chords intersect if exactly one endpoint of each lies in the other's arc
    return (c_in_ab ^ d_in_ab) and (a_in_cd ^ b_in_cd)

n = int(input())
chords = []
for _ in range(n):
    a, b = map(int, input().split())
    chords.append((a, b))

# Check all pairs of chords
found_intersection = False
for i in range(n):
    for j in range(i + 1, n):
        a1, b1 = chords[i]
        a2, b2 = chords[j]
        if intersect(a1, b1, a2, b2, 2 * n):
            found_intersection = True
            break
    if found_intersection:
        break

if found_intersection:
    print("Yes")
else:
    print("No")