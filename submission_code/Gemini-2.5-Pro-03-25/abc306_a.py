import sys

def solve():
    # Read the integer N
    n = int(sys.stdin.readline())
    
    # Read the string S and remove any trailing newline characters
    s = sys.stdin.readline().strip()
    
    # Initialize an empty list to store the characters of the result string
    result_chars = []
    
    # Iterate through each character in the input string S
    for char in s:
        # Append the character twice to the list
        result_chars.append(char)
        result_chars.append(char)
        # Alternatively, you could do: result_chars.append(char * 2)
        
    # Join the characters in the list to form the final string
    result_string = "".join(result_chars)
    
    # Print the final string to standard output
    print(result_string)

# Call the solve function to execute the logic
solve()