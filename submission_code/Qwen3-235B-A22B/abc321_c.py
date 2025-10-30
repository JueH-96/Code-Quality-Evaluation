def main():
    import sys
    K = int(sys.stdin.readline())
    nums = []
    # Iterate through all possible masks from 1 to 2^10 - 1
    for mask in range(1, 1 << 10):
        if mask == 1:  # Skip the subset {0}
            continue
        digits = []
        for d in range(10):
            if (mask >> d) & 1:
                digits.append(d)
        # Sort digits in decreasing order
        digits.sort(reverse=True)
        # Convert to number
        num_str = ''.join(map(str, digits))
        num = int(num_str)
        nums.append(num)
    # Sort all generated numbers
    nums.sort()
    # Output the K-th number (1-based index)
    print(nums[K-1])

if __name__ == "__main__":
    main()