def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    K = int(data[0])
    
    # Every 321-like number is created by choosing a non-empty subset of digits {0,1,...,9}
    # (except the case where the only chosen digit is 0, since 0 is not a positive integer)
    # and writing the digits in strictly decreasing order.
    numbers = []
    for mask in range(1, 1 << 10):
        digits = []
        for d in range(10):
            if mask & (1 << d):
                digits.append(d)
        # Exclude the subset that contains only 0
        if max(digits) == 0:
            continue
        # Sort the chosen digits in descending order to form a 321-like number.
        digits.sort(reverse=True)
        number = 0
        for d in digits:
            number = number * 10 + d
        numbers.append(number)
    
    # Sort all generated numbers in increasing order.
    numbers.sort()
    # Print the K-th smallest 321-like number (1-indexed).
    sys.stdout.write(str(numbers[K - 1]))
    
# Make sure to call main() so the program executes.
if __name__ == '__main__':
    main()