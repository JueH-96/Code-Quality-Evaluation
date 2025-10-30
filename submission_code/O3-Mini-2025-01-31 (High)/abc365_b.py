def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    n = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # Create a list of indices from 0 to n-1, sort by corresponding value in A.
    # The second largest will be at the second-to-last index in this sorted list.
    sorted_indices = sorted(range(n), key=lambda i: A[i])
    second_largest_index = sorted_indices[-2]  # 0-indexed position of the second largest element
    
    # Output the index in 1-indexed format
    print(second_largest_index + 1)

if __name__ == '__main__':
    main()