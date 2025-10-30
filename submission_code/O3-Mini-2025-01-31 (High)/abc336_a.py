def main():
    import sys
    # Read input from standard input (stdin)
    data = sys.stdin.read().split()
    if not data:
        return
    # Convert input to integer N
    N = int(data[0])
    # Construct the Dragon String: one "L", N occurrences of "o", then "n", "g"
    dragon_string = "L" + ("o" * N) + "ng"
    # Print the result to standard output (stdout)
    sys.stdout.write(dragon_string)

if __name__ == "__main__":
    main()