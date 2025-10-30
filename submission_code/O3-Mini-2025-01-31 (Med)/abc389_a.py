def main():
    import sys
    # Read the input string and strip any whitespace
    S = sys.stdin.read().strip()
    # Extract the first and third characters as integers
    num1 = int(S[0])
    num2 = int(S[2])
    # Calculate the product and print it
    print(num1 * num2)

if __name__ == '__main__':
    main()