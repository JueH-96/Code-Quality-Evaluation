def main():
    N = input().strip()
    # Extract digits
    a, b, c = N[0], N[1], N[2]
    # Form new integers
    bc_a = b + c + a
    c_ab = c + a + b
    # Print results
    print(int(bc_a), int(c_ab))

if __name__ == "__main__":
    main()