import sys

def solve():
    N = int(sys.stdin.readline())

    # This dictionary will store the minimum deliciousness found for each color.
    # The keys are bean colors (C_i), and the values are the smallest A_i
    # encountered so far for that specific color.
    min_deliciousness_by_color = {}

    for _ in range(N):
        # Read A_i and C_i for the current bean.
        # map(int, ...) converts the space-separated string parts to integers.
        A, C = map(int, sys.stdin.readline().split())

        # If this color (C) has been seen before:
        if C in min_deliciousness_by_color:
            # Update the minimum deliciousness for this color
            # if the current bean's deliciousness (A) is smaller.
            min_deliciousness_by_color[C] = min(min_deliciousness_by_color[C], A)
        else:
            # If this is the first time we see this color,
            # its deliciousness is currently the minimum for this color.
            min_deliciousness_by_color[C] = A

    # After processing all N beans, min_deliciousness_by_color will contain
    # the minimum deliciousness for each unique color that appeared in the input.

    # To maximize the minimum possible deliciousness of the bean eaten,
    # we need to choose the color for which its own minimum deliciousness is highest.
    # So, we find the maximum value among all the values (minimum deliciousness for each color)
    # stored in our dictionary.
    # The problem constraints guarantee N >= 1, so the dictionary will not be empty,
    # and max() will work correctly.
    result = max(min_deliciousness_by_color.values())

    # Print the final result.
    print(result)

# Call the solve function to execute the program.
if __name__ == '__main__':
    solve()