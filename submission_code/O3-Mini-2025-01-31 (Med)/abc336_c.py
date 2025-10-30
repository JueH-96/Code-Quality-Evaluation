def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read N from input
    N = int(data[0])
    
    # The sequence of good integers (which only contain even digits) corresponds
    # to taking a number in base-5 and mapping its digits d to 2*d,
    # with the sequence starting with 0. That is:
    # 0 (base-5: 0) -> 0, 1 (base-5: 1) -> 2, 2 -> 4, 3 -> 6, 4 -> 8, 10 -> 20, etc.
    # Since the sequence is 0-indexed (first element is 0), for input N (which is 1-indexed)
    # we convert (N-1) to base-5 and then perform the digit mapping.
    
    index = N - 1  # 0-indexed position
    
    # Convert 'index' to its base-5 representation.
    if index == 0:
        # Special case: when index is 0, the representation is simply 0.
        sys.stdout.write("0")
        return
    
    base5_digits = []
    while index:
        base5_digits.append(index % 5)
        index //= 5
    # The digits are currently in reverse order.
    base5_digits.reverse()
    
    # Map each base-5 digit d to 2*d to get the corresponding good integer digits.
    result = ''.join(str(2 * d) for d in base5_digits)
    
    sys.stdout.write(result)

if __name__ == "__main__":
    main()