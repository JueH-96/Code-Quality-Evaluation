def count_black_cells(A, B, C, D):
    def count_black(x1, y1, x2, y2):
        def count_in_range(a, b, c):
            return (b // c) - ((a - 1) // c)

        black_count = 0

        # Count cells in x = n (where n is an integer)
        black_count += count_in_range(x1, x2, 1)

        # Count cells in y = n (where n is an even number)
        black_count += count_in_range(y1, y2, 2)

        # Count cells in x + y = n (where n is an even number)
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                if (x + y) % 2 == 0:
                    black_count += 1

        return black_count

    # Calculate the total number of black cells in the rectangle
    total_black_cells = count_black(A, B, C, D)

    # The area of the black regions is twice the number of black cells
    return total_black_cells * 2

# Read input
import sys
input = sys.stdin.read
data = input().split()

A = int(data[0])
B = int(data[1])
C = int(data[2])
D = int(data[3])

# Calculate and print the result
result = count_black_cells(A, B, C, D)
print(result)