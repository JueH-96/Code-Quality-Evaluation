def main():
    import sys

    # Read exactly eight integers (they may be separated by spaces and/or newlines)
    numbers = list(map(int, sys.stdin.read().strip().split()))
    
    # Ensure we actually have 8 numbers; otherwise default to "No"
    if len(numbers) != 8:
        print("No")
        return
    
    # Condition 1: monotonically non-decreasing
    non_decreasing = all(numbers[i] <= numbers[i + 1] for i in range(7))
    
    # Condition 2: all between 100 and 675 inclusive
    in_range = all(100 <= x <= 675 for x in numbers)
    
    # Condition 3: all multiples of 25
    multiples_of_25 = all(x % 25 == 0 for x in numbers)
    
    # Output result
    if non_decreasing and in_range and multiples_of_25:
        print("Yes")
    else:
        print("No")


# Run the program
if __name__ == "__main__":
    main()