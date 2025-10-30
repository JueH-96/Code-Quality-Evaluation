def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    K = int(input_data[0])
    
    nums = []
    # There are 10 digits: 0 to 9.
    # We'll generate all non-empty subsets represented by bit masks from 1 to (1<<10)-1.
    # For each subset, skip the one that has single element [0] because that's 0 (not allowed).
    # Then, sort the digits in descending order to form our number.
    for mask in range(1, 1 << 10):
        digits = []
        for i in range(10):
            if mask & (1 << i):
                digits.append(i)
        # Skip the subset if the maximum digit is 0 (i.e., the subset is {0})
        if max(digits) == 0:
            continue
        # Arrange in strictly descending order (largest to smallest)
        digits_sorted_desc = sorted(digits, reverse=True)
        # Build the number from digits
        number = 0
        for d in digits_sorted_desc:
            number = number * 10 + d
        nums.append(number)
    
    # Sort the numbers in ascending order
    nums.sort()
    
    # Output the K-th smallest, note: K is 1-indexed
    sys.stdout.write(str(nums[K - 1]))

if __name__ == '__main__':
    main()