def solve():
    import sys

    # Read 64 integers (0 or 1) from stdin
    bits = list(map(int, sys.stdin.read().strip().split()))

    # Calculate the integer value
    result = 0
    for i, bit in enumerate(bits):
        result += bit << i

    # Print the result
    print(result)

def main():
    solve()

# When submitting, only the solve() function and its call are typically required.
# But including main() in case it's needed for a local test.
if __name__ == "__main__":
    main()