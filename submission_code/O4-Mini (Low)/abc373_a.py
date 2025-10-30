def main():
    import sys
    data = sys.stdin.read().splitlines()
    # We expect exactly 12 strings, one per line.
    count = 0
    for i, s in enumerate(data, start=1):
        if i > 12:
            break
        if len(s) == i:
            count += 1
    print(count)

if __name__ == "__main__":
    main()