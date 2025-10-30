def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # The key observation is that every good integer (i.e., an integer
    # whose all digits are even) can be uniquely produced by taking
    # a non-negative integer, writing it in base-5, and then replacing
    # each base-5 digit d with 2*d. For example, 
    #   0 in base-5 -> "0"       gives 0 (good integer)
    #   1 in base-5 -> "1"       gives 2
    #   2 in base-5 -> "2"       gives 4
    #   3 in base-5 -> "3"       gives 6
    #   4 in base-5 -> "4"       gives 8
    #   5 in base-5 -> "10"      gives 20
    # And the sequence in increasing order becomes:
    #   0, 2, 4, 6, 8, 20, 22, 24, 26, 28, 40, 42, ...
    #
    # Notice the 1st smallest is 0. So, if we let x = N-1 and write it in base-5,
    # then mapping each digit d to 2*d produces exactly the N-th smallest good integer.
    
    x = N - 1
    # Special handling: the base-5 representation of 0 should be "0"
    if x == 0:
        sys.stdout.write("0")
        return
    
    base5_digits = []
    while x:
        base5_digits.append(x % 5)
        x //= 5
    base5_digits.reverse()
    
    # Map each base-5 digit to its corresponding even digit.
    mapping = ['0', '2', '4', '6', '8']
    result = "".join(mapping[d] for d in base5_digits)
    sys.stdout.write(result)
    
if __name__ == "__main__":
    main()