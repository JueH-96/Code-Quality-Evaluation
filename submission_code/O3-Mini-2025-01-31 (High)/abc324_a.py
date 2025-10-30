def main():
    # Read the number of integers (not actually needed for the check)
    n = int(input().strip())
    
    # Read the list of numbers
    numbers = list(map(int, input().split()))
    
    # Check if all numbers are equal by converting to a set
    if len(set(numbers)) == 1:
        print("Yes")
    else:
        print("No")

# Call the main function
if __name__ == '__main__':
    main()