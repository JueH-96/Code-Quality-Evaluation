# YOUR CODE HERE
def main():
    import sys
    X = int(sys.stdin.readline().strip())
    # The ceiling of X/10 can be computed using integer arithmetic as (X + 9) // 10
    # This works correctly for both positive and negative X.
    print((X + 9) // 10)

main()