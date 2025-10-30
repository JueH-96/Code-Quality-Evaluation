def main():
    import sys

    # Read the two results
    S1, S2 = sys.stdin.read().strip().split()

    if S1 == 'sick':
        if S2 == 'sick':
            # Both got sick -> common oyster 1 is bad
            print(1)
        else:  # S2 == 'fine'
            # Only Takahashi got sick -> oyster 2 is bad
            print(2)
    else:  # S1 == 'fine'
        if S2 == 'sick':
            # Only Aoki got sick -> oyster 3 is bad
            print(3)
        else:  # S2 == 'fine'
            # Nobody got sick -> oyster 4 is bad
            print(4)

if __name__ == "__main__":
    main()