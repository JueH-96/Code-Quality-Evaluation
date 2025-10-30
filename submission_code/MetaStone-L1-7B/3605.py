def minBitArray(nums):
    result = []
    for x in nums:
        if (x & (x + 1)) == 0:
            a = x >> 1
            result.append(a)
            continue
        mask = x & (-x)
        if ((x ^ mask) & (mask - 1)) == 0:
            a = x - mask
            result.append(a)
        else:
            result.append(-1)
    return result