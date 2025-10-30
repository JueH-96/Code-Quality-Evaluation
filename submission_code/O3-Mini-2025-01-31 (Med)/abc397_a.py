def main():
    import sys
    # Read the single-line input as a string, then convert to float
    x = float(sys.stdin.read().strip())
    
    # Check the classification using the given temperature ranges.
    if x >= 38.0:
        print(1)
    elif x >= 37.5:
        print(2)
    else:
        print(3)

# Call the main function to execute the program.
if __name__ == '__main__':
    main()