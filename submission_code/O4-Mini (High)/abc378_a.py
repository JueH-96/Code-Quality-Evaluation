def main():
    import sys
    data = list(map(int, sys.stdin.read().split()))
    # Count occurrences of each color
    from collections import Counter
    cnt = Counter(data)
    # For each color, we can discard floor(count/2) pairs
    result = sum(v // 2 for v in cnt.values())
    print(result)

if __name__ == "__main__":
    main()