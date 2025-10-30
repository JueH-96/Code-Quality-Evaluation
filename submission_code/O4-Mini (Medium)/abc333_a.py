def main():
    # Read the input as a string (the digit itself)
    digit_str = input().strip()
    # Convert to int to know how many times to repeat
    count = int(digit_str)
    # Print the digit repeated count times
    print(digit_str * count)

if __name__ == "__main__":
    main()