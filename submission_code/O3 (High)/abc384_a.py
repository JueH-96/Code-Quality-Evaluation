def main():
    import sys

    # Read first line and split into components
    parts = sys.stdin.readline().strip().split()
    if len(parts) != 3:
        return  # Invalid input, but per constraints this shouldn't happen
    
    N = int(parts[0])           # Length of the string (not strictly needed)
    c1 = parts[1]               # Character to keep
    c2 = parts[2]               # Replacement character
    
    # Read the string S
    S = sys.stdin.readline().strip()
    
    # Generate the new string according to the rule
    result = ''.join(c if c == c1 else c2 for c in S)
    
    # Output the result
    print(result)

if __name__ == "__main__":
    main()