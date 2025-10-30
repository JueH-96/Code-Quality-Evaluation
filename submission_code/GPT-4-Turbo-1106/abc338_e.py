def does_chord_intersect(N, chords):
    # Sort the chords so that the first point is always less than the second point
    for i in range(N):
        chords[i] = sorted(chords[i])

    # Sort the chords based on their first point
    chords.sort(key=lambda x: x[0])

    # Use a stack to keep track of the chords
    stack = []
    for chord in chords:
        while stack and stack[-1] < chord[1]:
            stack.pop()
        stack.append(chord[1])

    # If the stack size is less than N, it means there was an intersection
    return len(stack) < N

# Read input from stdin
N = int(input().strip())
chords = [list(map(int, input().strip().split())) for _ in range(N)]

# Check for intersection and print the result
print("Yes" if does_chord_intersect(N, chords) else "No")