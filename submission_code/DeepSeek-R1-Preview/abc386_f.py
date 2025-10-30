k = int(input())
s = input().strip()
t = input().strip()

m = len(s)
n = len(t)

if abs(m - n) > k:
    print("No")
else:
    # Swap to make s the shorter string
    if m > n:
        s, t = t, s
        m, n = n, m

    # Initialize previous row
    prev_row = list(range(n + 1))

    for i in range(1, m + 1):
        # Initialize current row with infinity, except for the 0th position
        curr_row = [float('inf')] * (n + 1)
        curr_row[0] = i

        # Determine the window of j to consider
        min_j = max(1, i - k)
        max_j = min(n, i + k)

        for j in range(min_j, max_j + 1):
            if s[i-1] == t[j-1]:
                cost = 0
            else:
                cost = 1

            # Compute the three possible options
            option1 = prev_row[j] + 1    # delete
            option2 = curr_row[j-1] + 1  # insert
            option3 = prev_row[j-1] + cost  # replace or match

            curr_row[j] = min(option1, option2, option3)

        # Update previous row for next iteration
        prev_row = curr_row

    # Check if the edit distance is within the allowed K operations
    if prev_row[n] <= k:
        print("Yes")
    else:
        print("No")