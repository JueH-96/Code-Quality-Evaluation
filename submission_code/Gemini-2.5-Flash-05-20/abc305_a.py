import sys

def solve():
    # Read the input integer N from standard input.
    N = int(sys.stdin.readline())

    # Water stations are located at multiples of 5 km, including 0 km (start) and 100 km (goal).
    # This means the water station positions are 0, 5, 10, ..., 95, 100.

    # To find the nearest water station to Takahashi's position N,
    # we can divide N by 5, round the result to the nearest integer,
    # and then multiply by 5. This will give us the closest multiple of 5.

    # Example 1: N = 53
    # N / 5 = 10.6
    # round(10.6) = 11
    # 11 * 5 = 55
    # The nearest water station is at 55 km.

    # Example 2: N = 21
    # N / 5 = 4.2
    # round(4.2) = 4
    # 4 * 5 = 20
    # The nearest water station is at 20 km.

    # Example 3: N = 100
    # N / 5 = 20.0
    # round(20.0) = 20
    # 20 * 5 = 100
    # The nearest water station is at 100 km.

    # Python's built-in round() function works as follows for positive numbers:
    # - It rounds to the nearest integer.
    # - If a number is exactly halfway between two integers (e.g., 2.5), it rounds to the nearest even integer.
    #   (e.g., round(2.5) is 2, round(3.5) is 4). However, for N/5, we will never encounter
    #   a `.5` decimal for integer N values because water stations are spaced by 5km.
    #   N/5 will result in values like X.0, X.2, X.4, X.6, X.8.
    # - round(X.0) -> X
    # - round(X.2) -> X (for N % 5 = 1)
    # - round(X.4) -> X (for N % 5 = 2)
    # - round(X.6) -> X+1 (for N % 5 = 3)
    # - round(X.8) -> X+1 (for N % 5 = 4)
    # This behavior precisely matches the requirement to find the closest multiple of 5.
    # For N % 5 values of 0, 1, or 2, it rounds down to the nearest multiple of 5 less than or equal to N.
    # For N % 5 values of 3 or 4, it rounds up to the nearest multiple of 5 greater than or equal to N.

    nearest_station_position = round(N / 5) * 5

    # Print the result to standard output.
    sys.stdout.write(str(nearest_station_position) + "
")

# Call the solve function to execute the program.
solve()