def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    D = [int(x) for x in data[1:]]

    def is_rep_digit(month, day):
        # Combine the decimal string representations of month and day
        s = str(month) + str(day)
        # Check if all characters are the same
        return all(ch == s[0] for ch in s)

    count_rep_digit = 0
    # Iterate through all months and valid days
    for i in range(1, N + 1):
        for j in range(1, D[i - 1] + 1):
            if is_rep_digit(i, j):
                count_rep_digit += 1

    print(count_rep_digit)

# Call solve() after defining it
solve()