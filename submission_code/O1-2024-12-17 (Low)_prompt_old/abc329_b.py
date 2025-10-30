def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Find the maximum value in A
    maximum_value = max(A)

    # Remove all occurrences of the maximum value
    filtered = [x for x in A if x != maximum_value]

    # The answer is the maximum of the filtered list
    print(max(filtered))

# Call the solve function
solve()