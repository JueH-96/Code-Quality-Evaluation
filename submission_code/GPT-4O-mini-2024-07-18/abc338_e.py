def has_intersection(N, chords):
    # Normalize the chords to ensure A_i < B_i
    normalized_chords = [(min(a, b), max(a, b)) for a, b in chords]
    
    # Sort chords by their first point
    normalized_chords.sort()
    
    # Check for intersections
    for i in range(N):
        for j in range(i + 1, N):
            a1, b1 = normalized_chords[i]
            a2, b2 = normalized_chords[j]
            # Check if they intersect
            if (a1 < a2 < b1) or (a1 < b2 < b1):
                return "Yes"
    
    return "No"

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N = int(data[0])
    chords = [tuple(map(int, line.split())) for line in data[1:N+1]]
    result = has_intersection(N, chords)
    print(result)

if __name__ == "__main__":
    main()