def main():
    N = input().strip()
    # Extract the digits
    a, b, c = N[0], N[1], N[2]
    # Form the new numbers
    bca = int(b + c + a)
    cab = int(c + a + b)
    # Print the result
    print(bca, cab)

main()