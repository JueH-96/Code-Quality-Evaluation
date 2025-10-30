def main():
    import sys
    from itertools import combinations
    
    input_data = sys.stdin.read().strip()
    K = int(input_data)
    
    digits = list(range(10))
    nums = []
    
    # Generate all strictly decreasing digit combinations (subsets) except the empty set and {0} alone.
    for length in range(1, 11):
        for combo in combinations(digits, length):
            # Exclude the single subset {0}
            if length == 1 and combo[0] == 0:
                continue
            
            # Form the number by arranging in strictly decreasing order
            sorted_digits = sorted(combo, reverse=True)
            num = 0
            for d in sorted_digits:
                num = num * 10 + d
            nums.append(num)
    
    # Sort all generated 321-like numbers
    nums.sort()
    
    # Output the K-th smallest 321-like number
    print(nums[K-1])

# Do not forget to call main()
main()