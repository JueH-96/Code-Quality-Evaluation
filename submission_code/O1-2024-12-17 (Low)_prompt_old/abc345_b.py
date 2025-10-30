def solve():
    import sys
    X = int(sys.stdin.readline().strip())
    # Compute ceil(X/10). A common integer trick for ceiling division by a positive divisor d is:
    # ceil(X / d) = -( (-X) // d ).
    # This works correctly for both positive and negative X:
    result = -((-X) // 10)
    print(result)

def main():
    solve()

if __name__ == "__main__":
    main()