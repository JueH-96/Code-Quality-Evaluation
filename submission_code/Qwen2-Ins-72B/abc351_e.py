from bisect import bisect_left, bisect_right

def solve(N, points):
    # Sort points by their X-Y values
    points.sort(key=lambda p: p[0] - p[1])

    # Initialize variables
    even_x = []
    even_y = []
    odd_x = []
    odd_y = []
    ans = 0

    for x, y in points:
        # Calculate the distance to the nearest point with the same parity
        if x % 2 == 0:
            ans += min(bisect_right(even_x, x) - bisect_left(even_y, y),
                       bisect_right(even_y, y) - bisect_left(even_x, x))
            even_x.append(x)
            even_y.append(y)
        else:
            ans += min(bisect_right(odd_x, x) - bisect_left(odd_y, y),
                       bisect_right(odd_y, y) - bisect_left(odd_x, x))
            odd_x.append(x)
            odd_y.append(y)

    return ans

# Read input
N = int(input())
points = [list(map(int, input().split())) for _ in range(N)]

# Solve and print the answer
print(solve(N, points))