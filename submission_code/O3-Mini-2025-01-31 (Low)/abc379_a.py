def main():
    # Read the input integer as a string for easy slicing
    N = input().strip()
    
    # Extract digits: a is hundreds, b is tens, c is ones.
    a, b, c = N[0], N[1], N[2]
    
    # Form the two numbers:
    # first number: arrangement: b, c, a
    # second number: arrangement: c, a, b
    num1 = int(b + c + a)
    num2 = int(c + a + b)
    
    # Print the two numbers separated by a space
    print(num1, num2)

if __name__ == "__main__":
    main()