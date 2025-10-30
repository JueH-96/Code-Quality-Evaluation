def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # Initialize largest and second largest values, and their positions (1-indexed).
    largest = -1
    largest_index = -1
    second_largest = -1
    second_index = -1
    
    for i, val in enumerate(A, start=1):
        if val > largest:
            # Update second largest as the previous largest.
            second_largest = largest
            second_index = largest_index
            largest = val
            largest_index = i
        elif val > second_largest:
            second_largest = val
            second_index = i
    
    print(second_index)

if __name__ == '__main__':
    main()