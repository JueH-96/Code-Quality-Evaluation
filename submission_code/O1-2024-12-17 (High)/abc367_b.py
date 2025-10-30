def main():
    import sys
    from decimal import Decimal

    X_str = sys.stdin.readline().strip()
    X_dec = Decimal(X_str).normalize()
    # Just printing X_dec will remove unnecessary trailing zeros
    # and decimal points under these constraints.
    print(X_dec)

# Don't forget to call main()
if __name__ == "__main__":
    main()