def main():
    import sys
    # Read input value from stdin and remove any extra whitespace.
    data = sys.stdin.read().strip()
    if not data:
        return
    X = int(data)
    
    # Start with 1! = 1.
    fact = 1
    n = 1
    
    # Multiply sequentially until we reach X.
    while fact < X:
        n += 1
        fact *= n
    
    # At this point, fact equals X (as guaranteed by the problem).
    print(n)

# Call main() to execute the solution.
if __name__ == "__main__":
    main()