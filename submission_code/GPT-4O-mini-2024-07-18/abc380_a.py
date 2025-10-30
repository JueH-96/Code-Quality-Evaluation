def check_conditions(N):
    # Convert the number to a string to easily access each digit
    str_N = str(N)
    
    # Count the occurrences of each digit
    count_1 = str_N.count('1')
    count_2 = str_N.count('2')
    count_3 = str_N.count('3')
    
    # Check the conditions
    if count_1 == 1 and count_2 == 2 and count_3 == 3:
        return "Yes"
    else:
        return "No"

# Read input from standard input
import sys
input = sys.stdin.read
N = int(input().strip())

# Output the result
print(check_conditions(N))