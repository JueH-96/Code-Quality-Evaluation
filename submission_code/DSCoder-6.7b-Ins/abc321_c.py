def find_kth_smallest_321_number(k):
    # Initialize the first 321-like number
    num = [1]
    # Initialize the count of 321-like numbers
    count = 1
    # While the count of 321-like numbers is less than k
    while count < k:
        # Increment the last digit of the number
        num[-1] += 1
        # If the last digit is 10, append a 1 to the end of the number
        if num[-1] == 10:
            num.append(1)
        # Else, if the last digit is greater than the next digit, remove the last digit
        elif num[-1] > num[-2]:
            num = num[:-1]
            # If the last digit is 0, remove the last digit and decrement the second last digit
            if num[-1] == 0:
                num[-2] -= 1
                num = num[:-1]
        # Increment the count of 321-like numbers
        count += 1
    # Return the number as an integer
    return int(''.join(map(str, num)))

# Read the input from stdin
k = int(input())
# Find the kth smallest 321-like number
result = find_kth_smallest_321_number(k)
# Write the result to stdout
print(result)