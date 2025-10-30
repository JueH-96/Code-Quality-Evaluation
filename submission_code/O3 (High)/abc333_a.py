def main():
    import sys

    # Read the single integer N (strip to remove trailing newline/spaces)
    n_line = sys.stdin.readline().strip()
    if not n_line:
        return  # No input; nothing to do (defensive, though per problem N is provided)
    
    # Convert to integer
    try:
        N = int(n_line)
    except ValueError:
        # If somehow the input isn't an integer we just exit (problem guarantees integer input)
        return
    
    # Construct and print the string consisting of N copies of the digit N
    result = str(N) * N
    print(result)


# Entry point
if __name__ == "__main__":
    main()