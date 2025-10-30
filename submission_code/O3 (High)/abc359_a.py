def main():
    import sys

    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    n = int(data[0])
    # Count occurrences of "Takahashi" in the next n lines
    count = sum(1 for s in data[1:n+1] if s.strip() == "Takahashi")
    print(count)

if __name__ == "__main__":
    main()