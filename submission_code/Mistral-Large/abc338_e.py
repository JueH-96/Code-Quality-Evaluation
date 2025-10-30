import sys
input = sys.stdin.read

def find_intersections(N, chords):
    # Sort chords based on the first point
    chords.sort(key=lambda x: x[0])

    # Check for intersections
    for i in range(N):
        for j in range(i + 1, N):
            a1, b1 = chords[i]
            a2, b2 = chords[j]
            # Check if chords intersect using the circle property
            if (a1 < a2 < b1 < b2) or (a2 < a1 < b2 < b1):
                return "Yes"
    return "No"

def main():
    data = input().split()
    N = int(data[0])
    chords = []
    index = 1
    for i in range(N):
        A_i = int(data[index])
        B_i = int(data[index + 1])
        chords.append((A_i, B_i))
        index += 2

    result = find_intersections(N, chords)
    print(result)

if __name__ == "__main__":
    main()