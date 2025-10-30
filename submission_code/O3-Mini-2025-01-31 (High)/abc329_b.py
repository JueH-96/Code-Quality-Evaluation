def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # First value is N, subsequent are the list of numbers.
    n = int(input_data[0])
    numbers = list(map(int, input_data[1:]))
    
    # Find the maximum number in the list.
    max_number = max(numbers)
    
    # Filter out the maximum number to get only those that are not the largest.
    # There is guaranteed to be at least one number that is not the largest.
    other_numbers = [num for num in numbers if num != max_number]
    
    # Output the largest among the numbers that are not the maximum.
    print(max(other_numbers))

if __name__ == '__main__':
    main()