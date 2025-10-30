def main():
    import sys
    data = sys.stdin.read().split()
    # First token is N, the rest are the N strings
    # We only need to count how many are "Takahashi"
    count = sum(1 for s in data[1:] if s == "Takahashi")
    print(count)

if __name__ == "__main__":
    main()