def main():
    # Read the first line of input, which contains N and M
    N, M = map(int, input().split())

    # Read the second line of input, which contains the colors of the plates C_i
    C = input().split()

    # Read the third line of input, which contains the distinct colors D_i
    D = input().split()

    # Read the fourth line of input, which contains the prices P_i
    P = list(map(int, input().split()))

    # Create a dictionary to map plate colors to their prices
    color_to_price = {D[i]: P[i + 1] for i in range(M)}

    # Calculate the total price
    total_price = 0
    for color in C:
        # If the color is in the dictionary, add the corresponding price
        # Otherwise, add the default price P_0
        total_price += color_to_price.get(color, P[0])

    # Print the total price
    print(total_price)

if __name__ == "__main__":
    main()