def solve():
    import sys
    
    # Read input
    N = int(sys.stdin.readline().strip())
    
    # We want the N-th smallest "good integer".
    # Good integers have digits from {0,2,4,6,8} only.
    # The sequence of good integers in ascending order can be formed
    # by interpreting numbers in base 5 using digits {0,1,2,3,4}
    # then mapping those digits to {0,2,4,6,8}.
    #
    # 0 -> 0
    # 1 -> 2
    # 2 -> 4
    # 3 -> 6
    # 4 -> 8
    #
    # The 1st smallest good integer corresponds to 0 in base 5 => 0 in decimal.
    # So to get the N-th smallest, we convert N-1 to base 5, then map digits.

    # Special case: if N=1 => N-1=0 => base5 rep is "0" => maps to "0".
    # This approach works for all valid N.

    def to_base_5(num):
        # Convert num to a string in base 5
        if num == 0:
            return "0"
        digits = []
        while num > 0:
            digits.append(str(num % 5))
            num //=5
        return ''.join(digits[::-1])
    
    # Convert N-1 to base 5
    base5_str = to_base_5(N - 1)
    
    # Map the base5 digits to even digits
    mapping = {'0':'0','1':'2','2':'4','3':'6','4':'8'}
    result_str = ''.join(mapping[d] for d in base5_str)
    
    # Print the result
    print(result_str)

def main():
    solve()

if __name__ == "__main__":
    main()