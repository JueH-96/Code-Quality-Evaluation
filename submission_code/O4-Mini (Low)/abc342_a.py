def main():
    import sys

    S = sys.stdin.readline().strip()
    # Find the unique character (the one that appears exactly once)
    for ch in set(S):
        if S.count(ch) == 1:
            unique_char = ch
            break
    # Output its 1-based position
    print(S.index(unique_char) + 1)

if __name__ == "__main__":
    main()