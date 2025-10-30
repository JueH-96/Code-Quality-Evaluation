def main():
    import sys

    s1, s2 = sys.stdin.read().split()

    # Takahashi eats oysters 1 and 2
    # Aoki eats oysters 1 and 3
    for c in range(1, 5):
        # Determine Takahashi's result if oyster c is bad
        takahashi = "sick" if c in (1, 2) else "fine"
        # Determine Aoki's result if oyster c is bad
        aoki = "sick" if c in (1, 3) else "fine"

        if takahashi == s1 and aoki == s2:
            print(c)
            return

if __name__ == "__main__":
    main()