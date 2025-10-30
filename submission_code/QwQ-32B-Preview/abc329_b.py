def main():
    import sys

    # Read the number of elements
    N = int(sys.stdin.readline())

    # Read the list of integers
    A = list(map(int, sys.stdin.readline().split()))

    # Find the maximum value in the list
    max_value = max(A)

    # Create a list excluding all elements equal to max_value
    filtered_list = [x for x in A if x != max_value]

    # Find the maximum value in the filtered list
    second_max = max(filtered_list)

    # Print the result
    print(second_max)

if __name__ == "__main__":
    main()