def sumImbalanceNumbers(self, nums: List[int]) -> int:
    n = len(nums)
    total = 0

    # Precompute left_sorted: for each i, a list of tuples (value, index) sorted by value, containing elements before i
    left_sorted = []
    for i in range(n):
        current = []
        if i > 0:
            current = nums[:i]
        current_sorted = sorted(current)
        left_sorted.append(current_sorted)

    # Precompute right_sorted: for each j, a list of tuples (value, index) sorted by value, containing elements after j
    right_sorted = []
    for j in range(n):
        current = []
        if j < n - 1:
            current = nums[j+1:]
        current_sorted = sorted(current)
        right_sorted.append(current_sorted)

    for i in range(n):
        for j in range(i + 1, n):
            a = nums[i]
            b = nums[j]
            if b > a and (b - a) > 1:
                # Find L: first index l < i where nums[l] >= b
                if not left_sorted[i]:
                    L = 0
                else:
                    # Extract values and indices
                    values = [v for v, idx in left_sorted[i]]
                    # Find the first index in values >= b
                    pos = bisect.bisect_left(values, b)
                    if pos < len(values):
                        L = left_sorted[i][pos][1]
                    else:
                        L = 0

                # Find R: first index r > j where nums[r] <= a
                if not right_sorted[j]:
                    R = n
                else:
                    values = [v for v, idx in right_sorted[j]]
                    # Find the first index in values <= a
                    pos = bisect.bisect_right(values, a)
                    if pos > 0:
                        R = right_sorted[j][pos - 1][1]
                    else:
                        R = n

                left_choices = i - L + 1
                right_choices = R - j
                total += left_choices * right_choices

    return total