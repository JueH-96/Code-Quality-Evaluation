def main():
    import sys
    # Read input from standard input
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    A = int(input_data[0])
    B = int(input_data[1])
    
    # Calculate the number of attacks required using ceiling division
    attacks = (A + B - 1) // B
    
    # Output the result
    print(attacks)

# Call the main function
if __name__ == "__main__":
    main()