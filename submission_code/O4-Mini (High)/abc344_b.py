import sys

def main():
    arr = []
    # Read integers one per line until we read a 0 (inclusive).
    for line in sys.stdin:
        x = int(line.strip())
        arr.append(x)
        if x == 0:
            break

    # Print in reverse order, one per line.
    for v in reversed(arr):
        print(v)

# Call main to execute the solution.
if __name__ == "__main__":
    main()