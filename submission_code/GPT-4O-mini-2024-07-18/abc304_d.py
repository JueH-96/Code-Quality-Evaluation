def min_max_strawberries(W, H, strawberries, A, vertical_cuts, B, horizontal_cuts):
    # Create lists to hold the counts of strawberries in each segment
    vertical_segments = [0] * (A + 1)
    horizontal_segments = [0] * (B + 1)
    
    # Count strawberries in each vertical segment
    for p, q in strawberries:
        # Find the vertical segment for this strawberry
        for i in range(A):
            if vertical_cuts[i] > p:
                vertical_segments[i] += 1
                break
        else:
            vertical_segments[A] += 1  # Last segment

    # Count strawberries in each horizontal segment
    for p, q in strawberries:
        # Find the horizontal segment for this strawberry
        for j in range(B):
            if horizontal_cuts[j] > q:
                horizontal_segments[j] += 1
                break
        else:
            horizontal_segments[B] += 1  # Last segment

    # Now we need to calculate the number of strawberries in each piece
    piece_counts = []
    for v in vertical_segments:
        for h in horizontal_segments:
            piece_counts.append(v + h)

    # The minimum and maximum strawberries in any piece
    min_strawberries = min(piece_counts)
    max_strawberries = max(piece_counts)

    return min_strawberries, max_strawberries

import sys
input = sys.stdin.read
data = input().splitlines()

# Read the dimensions of the cake
W, H = map(int, data[0].split())
# Read the number of strawberries
N = int(data[1])
# Read the coordinates of the strawberries
strawberries = [tuple(map(int, data[i + 2].split())) for i in range(N)]
# Read the number of vertical cuts
A = int(data[N + 2])
# Read the vertical cuts
vertical_cuts = list(map(int, data[N + 3].split()))
# Read the number of horizontal cuts
B = int(data[N + 4])
# Read the horizontal cuts
horizontal_cuts = list(map(int, data[N + 5].split()))

# Get the minimum and maximum strawberries in a piece
m, M = min_max_strawberries(W, H, strawberries, A, vertical_cuts, B, horizontal_cuts)

# Print the result
print(m, M)