def main():
    import sys
    input_line = sys.stdin.readline().strip()
    # Read X and Y from input
    X, Y = map(int, input_line.split())
    
    # Calculate the difference
    diff = Y - X
    
    # Check if Takahashi uses the stairs based on the conditions:
    # Moving up using stairs if the upward difference is 2 floors or less.
    # Moving down using stairs if the downward difference is 3 floors or less.
    if (diff > 0 and diff <= 2) or (diff < 0 and abs(diff) <= 3):
        print("Yes")
    else:
        print("No")

# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()