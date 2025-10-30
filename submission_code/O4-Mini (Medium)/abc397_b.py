def main():
    S = input().strip()
    pos = 0        # current index in the infinite "ioioio..." pattern
    inserted = 0   # count of insertions made

    for c in S:
        # advance in the pattern until we match S's current character
        while True:
            # expected char at position pos in the pattern
            expected = 'i' if (pos % 2 == 0) else 'o'
            if expected == c:
                # match: consume this pattern position for the current S character
                pos += 1
                break
            else:
                # mismatch: we must insert the expected char here
                inserted += 1
                pos += 1

    # after matching all of S, ensure total length is even
    if pos % 2 == 1:
        inserted += 1
        pos += 1

    print(inserted)

if __name__ == "__main__":
    main()