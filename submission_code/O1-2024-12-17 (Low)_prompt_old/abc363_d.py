def solve():
    import sys

    # Read input
    N_str = sys.stdin.readline().strip()
    N = int(N_str)

    # If N <= 10, the answer is simply N-1 (the first 10 palindromes are 0..9).
    if N <= 10:
        print(N - 1)
        return

    # Precompute the cumulative counts T(d) of how many palindromes of up to d digits exist.
    # count_d(d) = number of d-digit palindromes.
    #  - d = 1 => 10 (0..9).
    #  - d >= 2 => 9 * 10^((d-1)//2).
    # T(d) = sum_{k=1..d} count_d(k).
    # We only need to go up to some reasonably large d since 10^18 palindromes occur well before d=120.
    # (This is because the counts grow exponentially.)
    def count_d(d):
        if d == 1:
            return 10
        return 9 * pow(10, (d - 1) // 2)

    max_d = 120  # More than enough for N <= 10^18
    T = [0] * (max_d + 1)
    for d in range(1, max_d + 1):
        T[d] = T[d - 1] + count_d(d)

    # Find digit length d such that T[d - 1] < N <= T[d]
    # T[d] is the total number of palindromes up to digit-length d.
    # We already handled N <= 10, so N > 10 here.
    # So we skip the first 10 one-digit palindromes by using T[1] = 10, etc.
    d_low, d_high = 1, max_d
    while d_low < d_high:
        mid = (d_low + d_high) // 2
        if T[mid] >= N:
            d_high = mid
        else:
            d_low = mid + 1
    d = d_low

    # offset = how many palindromes into the d-digit group
    # That is offset = N - T[d-1].
    offset = N - T[d - 1]

    # Construct the palindrome with digit length d using the offset.
    # For d=1, we'd just do offset-1, but here N>10 => d>=2. We handle it generally anyway.
    if d == 1:
        # The offset-th palindrome among the 1-digit group: 0..9
        # offset is 1-based, so offset-th => offset-1 in [0..9].
        print(offset - 1)
        return

    # The number of d-digit palindromes is count_d(d) = 9 * 10^((d-1)//2).
    # We'll find the "first half", then mirror.
    # If d is even, halfLen = d//2, smallestHalf = 10^(halfLen-1).
    # If d is odd, halfLen = (d+1)//2, smallestHalf = 10^(halfLen-1).
    half_len = (d + 1) // 2  # works for both even and odd
    smallest_half = pow(10, half_len - 1)  # e.g. d=2 => half_len=1 => smallest_half=1

    # The 1-based offset means half = smallest_half + (offset - 1)
    half_value = smallest_half + (offset - 1)

    half_str = str(half_value)
    if d % 2 == 0:
        # even-length palindrome => mirror the entire half
        # e.g. half=12 => palindrome= "12" + "21"
        palindrome_str = half_str + half_str[::-1]
    else:
        # odd-length palindrome => mirror all but the last character of half
        # e.g. half=123 => palindrome = "123" + "21" = "12321"
        palindrome_str = half_str + half_str[-2::-1]

    print(palindrome_str)

# Let's call solve() to complete the requirement.
solve()