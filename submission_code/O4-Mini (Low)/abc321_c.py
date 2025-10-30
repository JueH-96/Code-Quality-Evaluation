def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    K = int(data)
    nums = []
    # We represent subsets of digits {0,...,9} via bitmask from 1 to (1<<10)-1
    # Exclude the subset {0} (mask == 1<<0 == 1), since that yields the number 0 which is not positive.
    for mask in range(1, 1 << 10):
        if mask == 1:  # skip the subset containing only digit 0
            continue
        # collect digits in this subset
        digits = []
        for d in range(10):
            if (mask >> d) & 1:
                digits.append(d)
        # sort in strictly decreasing order
        digits.sort(reverse=True)
        # the leading digit won't be zero because mask != 1,
        # so there is at least one digit >= 1
        # form the integer
        val = 0
        for d in digits:
            val = val * 10 + d
        nums.append(val)
    # sort all 321-like numbers
    nums.sort()
    # K is 1-based
    print(nums[K-1])

if __name__ == "__main__":
    main()