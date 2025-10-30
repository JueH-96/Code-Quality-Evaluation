def main():
    # Read the input from stdin, which is expected in a single line "A B"
    a, b = map(int, input().split())
    # Compute the square of the sum (A+B)
    result = (a + b) ** 2
    # Output the result
    print(result)

# Call main to execute the solution
if __name__ == '__main__':
    main()