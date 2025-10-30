def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    # Map from base-5 digit to even-digit
    mapping = {0: '0', 1: '2', 2: '4', 3: '6', 4: '8'}
    
    # The sequence's first element when N = 1 is 0.
    # We can think of the sequence as the numbers represented by the base-5 representation of (N-1),
    # then mapping each base-5 digit, with mapping: 0->0, 1->2, 2->4, 3->6, 4->8.
    # For example, when N = 8, we have N-1 = 7, which in base 5 is "12". 
    # Map '1' -> '2' and '2' -> '4' so we get 24.
    
    # Convert (N - 1) into base-5 representation:
    num = N - 1
    if num == 0:
        # N - 1 = 0, meaning N = 1 so the result is 0.
        print(0)
        return
    base5_digits = []
    while num > 0:
        digit = num % 5
        base5_digits.append(digit)
        num //= 5
    # The digits are collected in reverse order, reverse to get correct order.
    base5_digits.reverse()
    
    # Map the digits to even numbers 
    result = ''.join(mapping[digit] for digit in base5_digits)
    
    # Print the final result
    sys.stdout.write(result)
    
if __name__ == "__main__":
    main()