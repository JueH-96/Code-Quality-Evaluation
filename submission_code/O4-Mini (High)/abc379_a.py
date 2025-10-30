def main():
    N = input().strip()
    # Extract digits
    a, b, c = N[0], N[1], N[2]
    # Form bca and cab
    first = int(b + c + a)
    second = int(c + a + b)
    # Print results
    print(first, second)

# Call main to execute
main()