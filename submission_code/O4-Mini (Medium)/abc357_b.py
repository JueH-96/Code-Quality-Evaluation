def main():
    import sys

    S = sys.stdin.readline().strip()
    upper_count = sum(1 for c in S if c.isupper())
    lower_count = len(S) - upper_count

    if upper_count > lower_count:
        print(S.upper())
    else:
        print(S.lower())

if __name__ == "__main__":
    main()