def main():
    import sys

    # Read input values
    R, G, B = map(int, sys.stdin.readline().split())
    C = sys.stdin.readline().strip()

    # Build a dictionary of colors to their prices
    prices = {
        "Red": R,
        "Green": G,
        "Blue": B
    }

    # Remove the disliked color
    del prices[C]

    # Print the minimum price among the remaining pens
    print(min(prices.values()))

if __name__ == "__main__":
    main()