def smallestString(s: str) -> str:
    n = len(s)
    next_a = [n] * n  # Initialize to n, which is beyond the last index
    last_a = n  # last_a is the next_a for current i

    for i in range(n-1, -1, -1):
        if s[i] == 'a':
            last_a = i
        next_a[i] = last_a

    # Find the earliest i where s[i] > 'a'
    earliest_i = None
    for i in range(n):
        if s[i] > 'a':
            earliest_i = i
            break

    if earliest_i is not None:
        # Find j
        j = next_a[earliest_i]
        if j < n:
            # substring is earliest_i to j-1
            start = earliest_i
            end = j
        else:
            # j is n, so substring is earliest_i to n-1
            start = earliest_i
            end = n
        # Modify the substring
        result = list(s)
        for k in range(start, end):
            result[k] = chr(ord(result[k]) - 1)
        return ''.join(result)
    else:
        # All characters are 'a's; modify the entire string
        result = list(s)
        for k in range(n):
            result[k] = chr(ord(result[k]) - 1)
        return ''.join(result)