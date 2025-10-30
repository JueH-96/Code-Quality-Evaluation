# YOUR CODE HERE
def check_conditions(s):
    # Check if all numbers are between 100 and 675
    if not all(100 <= num <= 675 for num in s):
        return "No"
    
    # Check if all numbers are multiples of 25
    if not all(num % 25 == 0 for num in s):
        return "No"
    
    # Check if the sequence is monotonically non-decreasing
    if not all(s[i] <= s[i+1] for i in range(len(s)-1)):
        return "No"
    
    return "Yes"

# Read input from stdin
s = list(map(int, input().split()))

# Print the answer to stdout
print(check_conditions(s))