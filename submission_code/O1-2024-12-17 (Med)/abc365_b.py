def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Sort indices by corresponding A-values in descending order
    sorted_indices = sorted(range(N), key=lambda i: A[i], reverse=True)
    # The second largest element's index in 1-based indexing
    second_largest_index = sorted_indices[1] + 1
    print(second_largest_index)

# Do not remove the call to main() below
main()