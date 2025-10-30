# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    D = int(input[1])
    points = [(int(input[2 + 2 * i]), int(input[3 + 2 * i])) for i in range(N)]

    # Transform points to (x+y, x-y) coordinates
    transformed = [(x + y, x - y) for x, y in points]
    transformed.sort()

    # Calculate prefix sums for the transformed coordinates
    prefix_sum_x_plus_y = [0] * (N + 1)
    prefix_sum_x_minus_y = [0] * (N + 1)
    for i in range(N):
        prefix_sum_x_plus_y[i + 1] = prefix_sum_x_plus_y[i] + transformed[i][0]
        prefix_sum_x_minus_y[i + 1] = prefix_sum_x_minus_y[i] + transformed[i][1]

    count = 0
    for x, y in transformed:
        # Calculate the range for x+y and x-y
        max_x_plus_y = x + y + D
        min_x_plus_y = x + y - D
        max_x_minus_y = x - y + D
        min_x_minus_y = x - y - D

        # Find the range of indices for x+y
        left_x_plus_y = bisect.bisect_left(transformed, (min_x_plus_y, float('-inf')))
        right_x_plus_y = bisect.bisect_right(transformed, (max_x_plus_y, float('inf')))

        # Find the range of indices for x-y
        left_x_minus_y = bisect.bisect_left(transformed, (float('-inf'), min_x_minus_y))
        right_x_minus_y = bisect.bisect_right(transformed, (float('inf'), max_x_minus_y))

        # Calculate the number of valid points
        valid_points = 0
        for i in range(left_x_plus_y, right_x_plus_y):
            if transformed[i][1] >= min_x_minus_y and transformed[i][1] <= max_x_minus_y:
                valid_points += 1

        count += valid_points

    # Each point is counted 4 times (once for each quadrant), so divide by 4
    print(count // 4)

if __name__ == "__main__":
    main()