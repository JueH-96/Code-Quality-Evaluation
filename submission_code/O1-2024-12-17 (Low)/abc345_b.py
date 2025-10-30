def main():
    import sys
    X = int(sys.stdin.readline().strip())
    # Compute the ceiling of X/10 using integer arithmetic.
    # A neat trick: ceil(X/10) = (X + 9) // 10 for all integer X.
    result = (X + 9) // 10
    print(result)

# Don't forget to call main!
main()