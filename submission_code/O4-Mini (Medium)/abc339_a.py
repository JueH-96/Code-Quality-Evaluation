def main():
    import sys
    S = sys.stdin.readline().rstrip()
    # Split on the last dot and take the suffix
    # rsplit with maxsplit=1 returns a list of at most 2 parts
    # The last element is what comes after the final '.'
    print(S.rsplit('.', 1)[-1])

if __name__ == "__main__":
    main()