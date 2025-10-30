import sys

def main() -> None:
    """
    Reads 12 strings from standard input and counts how many of them
    have a length equal to their 1-based position.
    """
    count = 0
    for i in range(1, 13):                     # positions 1 .. 12
        s = sys.stdin.readline().rstrip('
')  # read each string, strip newline only
        if len(s) == i:
            count += 1
    print(count)

if __name__ == "__main__":
    main()