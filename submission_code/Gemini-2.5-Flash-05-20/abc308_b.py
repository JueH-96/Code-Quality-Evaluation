import sys

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Read plate colors C
    C = sys.stdin.readline().split()

    # Read special colors D
    D = sys.stdin.readline().split()

    # Read prices P_0, P_1, ..., P_M
    P = list(map(int, sys.stdin.readline().split()))

    # P_0 is the default price
    P0 = P[0]

    # Create a dictionary for special prices
    # The prices for D_1 to D_M are P_1 to P_M respectively.
    # P[1:] contains P_1, P_2, ..., P_M
    special_prices = {}
    for i in range(M):
        special_prices[D[i]] = P[i+1]

    total_cost = 0

    # Calculate the total cost
    for plate_color in C:
        if plate_color in special_prices:
            total_cost += special_prices[plate_color]
        else:
            total_cost += P0

    # Print the total cost
    print(total_cost)

# Call the solve function to run the program
if __name__ == '__main__':
    solve()