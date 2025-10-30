import sys

def read_input():
    """Read input from stdin."""
    lines = sys.stdin.readlines()
    N, W = map(int, lines[0].split())
    items = []
    for line in lines[1:]:
        w, v = map(int, line.split())
        items.append((w, v))
    return N, W, items

def calculate_happiness(N, W, items):
    """Calculate the maximum total happiness."""
    # Initialize a list to store the maximum happiness for each weight
    dp = [0] * (W + 1)

    # Iterate over each item
    for w, v in items:
        # Iterate from W to w in reverse order
        for i in range(W, w - 1, -1):
            # Calculate the maximum happiness for the current weight
            for k in range(1, i // w + 1):
                dp[i] = max(dp[i], dp[i - k * w] + k * v - k ** 2)

    # Return the maximum happiness for the maximum weight
    return max(dp)

def main():
    """Main function."""
    N, W, items = read_input()
    happiness = calculate_happiness(N, W, items)
    print(happiness)

if __name__ == "__main__":
    main()