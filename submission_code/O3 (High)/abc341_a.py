def main():
    import sys

    # Read N
    n_line = sys.stdin.readline().strip()
    if not n_line:
        return
    N = int(n_line)

    # Build and output the alternating string: "10"*N followed by one extra "1"
    result = '10' * N + '1'
    print(result)


# Call the main function
if __name__ == "__main__":
    main()