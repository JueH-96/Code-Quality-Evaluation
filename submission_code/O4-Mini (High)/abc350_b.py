def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    # Initialize all holes with a tooth (True means tooth present)
    has_tooth = [True] * (n + 1)
    # Process each treatment: toggle the presence of a tooth
    for _ in range(q):
        t = int(next(it))
        has_tooth[t] = not has_tooth[t]
    # Count how many teeth remain
    print(sum(has_tooth[1:]))

if __name__ == "__main__":
    main()