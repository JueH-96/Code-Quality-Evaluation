def check_conditions(numbers):
    # Check if the sequence is monotonically non-decreasing
    if not all(numbers[i] <= numbers[i+1] for i in range(len(numbers)-1)):
        return False

    # Check if all numbers are between 100 and 675, inclusive
    if not all(100 <= num <= 675 for num in numbers):
        return False

    # Check if all numbers are multiples of 25
    if not all(num % 25 == 0 for num in numbers):
        return False

    return True

# Read input from stdin
numbers = list(map(int, input().split()))

# Check conditions and print result
if check_conditions(numbers):
    print("Yes")
else:
    print("No")