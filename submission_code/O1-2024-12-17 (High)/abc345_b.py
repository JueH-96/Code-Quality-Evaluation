def main():
    X = int(input().strip())
    # Compute ceil(X / 10) using integer arithmetic
    # A well-known trick for computing ceiling for a positive divisor (here 10)
    # is (X + (10 - 1)) // 10, i.e. (X + 9) // 10.
    # This works correctly for both positive and negative X.
    answer = (X + 9) // 10
    print(answer)

# Do not forget to call main()!
main()