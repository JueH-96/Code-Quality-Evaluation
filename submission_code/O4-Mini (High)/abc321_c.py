def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    K = int(data)
    nums = []
    # Each 321-like number corresponds to a nonempty subset of digits 0-9,
    # except the singleton {0}.  We generate all masks from 1 to (1<<10)-1,
    # skip mask==1 (which is {0}), and build the number by placing bits
    # in descending digit order.
    for mask in range(1, 1 << 10):
        if mask == 1:
            continue  # skip the subset {0}
        num = 0
        # Build digits from 9 down to 0
        for d in range(9, -1, -1):
            if (mask >> d) & 1:
                num = num * 10 + d
        nums.append(num)
    nums.sort()
    # K-th smallest (1-based)
    print(nums[K-1])

if __name__ == "__main__":
    main()