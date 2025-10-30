def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    K = int(data[0])
    
    # We can generate all 321-like numbers by choosing a non-empty subset
    # of digits from {0,1,2,...,9} such that the maximum digit (the first one in descending order)
    # is not 0. Then the 321-like number is just those digits arranged in descending order.
    numbers = []
    
    # There are 2^10 - 1 = 1023 possible subsets, so we can enumerate all using bit masks.
    # We'll check that the highest digit in the subset is not 0.
    for mask in range(1, 1 << 10):  # iterate from 1 to 1023
        digits = []
        for d in range(10):
            if mask & (1 << d):
                digits.append(d)
        # We want the digits in descending order
        digits.sort(reverse=True)
        # The first digit must be non-zero (ensured by checking if max digit is 0)
        # But if 0 is the only digit then it is 0, not a positive integer.
        # So we ensure that the first digit is non-zero:
        if digits[0] == 0:
            continue
        # Form the number from the digits
        num = 0
        for digit in digits:
            num = num * 10 + digit
        numbers.append(num)
        
    # Now sort the numbers in ascending order.
    numbers.sort()
    
    # Since K is 1-indexed, output the K-th smallest number
    # As guaranteed there are at least K numbers.
    print(numbers[K-1])

if __name__ == '__main__':
    main()