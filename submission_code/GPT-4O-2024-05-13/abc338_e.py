# YOUR CODE HERE
def do_chords_intersect(N, chords):
    # Normalize the chords such that A_i < B_i
    normalized_chords = [(min(a, b), max(a, b)) for a, b in chords]
    
    # Sort the chords by their starting point
    normalized_chords.sort()
    
    # Check for intersections
    for i in range(N):
        for j in range(i + 1, N):
            a1, b1 = normalized_chords[i]
            a2, b2 = normalized_chords[j]
            # Check if chord (a1, b1) intersects with chord (a2, b2)
            if a1 < a2 < b1 < b2 or a2 < a1 < b2 < b1:
                return "Yes"
    
    return "No"

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
chords = []
for i in range(N):
    A = int(data[2 * i + 1])
    B = int(data[2 * i + 2])
    chords.append((A, B))

print(do_chords_intersect(N, chords))