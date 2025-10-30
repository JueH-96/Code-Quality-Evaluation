def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    K = int(data)
    nums = []
    # We use a bitmask from 1 to (1<<10)-1, bits 0..9 represent digits 0..9
    # Skip mask == 1<<0 (i.e., mask == 1) because that would give the number "0"
    for mask in range(1, 1 << 10):
        if mask == 1:  # skip the subset {0}
            continue
        # build the number by taking digits in descending order 9..0
        # whenever the bit is set
        x = 0
        for d in range(9, -1, -1):
            if (mask >> d) & 1:
                x = x * 10 + d
        nums.append(x)
    # sort all these 321-like numbers
    nums.sort()
    # 1-based K
    result = nums[K-1]
    sys.stdout.write(str(result))

if __name__ == "__main__":
    main()