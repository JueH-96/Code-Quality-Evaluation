# YOUR CODE HERE
def replace_characters(N, c1, c2, S):
    # Create a new string with the replacements
    result = ''.join(c1 if char == c1 else c2 for char in S)
    return result

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read the first line for N, c1, c2
    first_line = data[0].split()
    N = int(first_line[0])
    c1 = first_line[1]
    c2 = first_line[2]
    
    # Read the second line for S
    S = data[1]
    
    # Get the result
    result = replace_characters(N, c1, c2, S)
    
    # Print the result
    print(result)