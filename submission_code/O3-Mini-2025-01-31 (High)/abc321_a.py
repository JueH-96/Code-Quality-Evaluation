def main():
    import sys
    # Read input and strip any surrounding whitespace
    data = sys.stdin.read().strip()
    if not data:
        return
    # Get the first token (should be the number as a string)
    s = data.split()[0]
    
    # Check the "321-like" property: digits must be strictly decreasing
    is_321_like = all(s[i] > s[i+1] for i in range(len(s) - 1))
    
    # Print Yes if the property holds, No otherwise
    sys.stdout.write("Yes" if is_321_like else "No")
    
# Call the main function to execute the program
main()