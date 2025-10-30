from collections import defaultdict

def can_place_piece(N, M, a, b):
    # Create a set of occupied squares
    occupied = set((a[i], b[i]) for i in range(M))

    # Check all squares for valid placement
    count = 0
    for i in range(1, N+1):
        for j in range(1, N+1):
            if (i, j) not in occupied:
                # Check if the square can be captured by any existing piece
                can_capture = False
                for k in range(M):
                    if (a[k], b[k]) in occupied:
                        if (i+2, j+1) == (a[k], b[k]) or \
                           (i+1, j+2) == (a[k], b[k]) or \
                           (i-1, j+2) == (a[k], b[k]) or \
                           (i-2, j+1) == (a[k], b[k]) or \
                           (i-2, j-1) == (a[k], b[k]) or \
                           (i-1, j-2) == (a[k], b[k]) or \
                           (i+1, j-2) == (a[k], b[k]) or \
                           (i+2, j-1) == (a[k], b[k]):
                            can_capture = True
                            break
                if not can_capture:
                    count += 1
    return count

# Read input
N, M = map(int, input().split())
a = []
b = []
for _ in range(M):
    x, y = map(int, input().split())
    a.append(x)
    b.append(y)

# Solve the problem
result = can_place_piece(N, M, a, b)
print(result)