def main():
    import sys
    data = sys.stdin.read().strip()
    N = int(data)
    
    # We want the N-th smallest "good integer", where "good integers"
    # are those whose decimal digits are all in {0,2,4,6,8}.
    #
    # If we list them in order:
    #   0, 2, 4, 6, 8, 20, 22, 24, ...
    # we see that this is exactly the sequence of nonnegative integers
    # when written in base-5 (digits 0 through 4), then each base-5 digit
    # d is mapped to 2*d in decimal (so 0->0,1->2,2->4,3->6,4->8).
    #
    # The N-th term in this list corresponds to (N-1) in base-5, with
    # that digit-to-digit mapping.
    
    x = N - 1
    # Convert x to base-5 representation (as digits 0..4)
    if x == 0:
        # The 1st good integer is 0
        print(0)
        return
    
    digits = []
    while x > 0:
        x, r = divmod(x, 5)
        digits.append(r)
    # digits now holds the base-5 digits in reverse order
    
    # Map base-5 digit -> even decimal digit
    # 0->0, 1->2, 2->4, 3->6, 4->8
    mapping = ['0', '2', '4', '6', '8']
    
    # Build the result string
    result_chars = []
    for d in reversed(digits):
        result_chars.append(mapping[d])
    
    # Join and print
    print(''.join(result_chars))


if __name__ == "__main__":
    main()