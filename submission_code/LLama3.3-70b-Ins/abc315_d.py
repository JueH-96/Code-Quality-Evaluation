def count_remaining_cookies():
    # Read the input from stdin
    H, W = map(int, input().split())
    cookies = [list(input()) for _ in range(H)]

    # Initialize a set to keep track of marked cookies
    marked = set()

    # Function to mark cookies in a row
    def mark_row(row):
        if len(set(cookies[row])) == 1 and len(cookies[row]) > 1:
            for col in range(W):
                marked.add((row, col))

    # Function to mark cookies in a column
    def mark_col(col):
        column = [cookies[row][col] for row in range(H)]
        if len(set(column)) == 1 and len(column) > 1:
            for row in range(H):
                marked.add((row, col))

    # Function to remove marked cookies
    def remove_marked():
        nonlocal cookies
        cookies = [[cookie if (row, col) not in marked else '.' for col, cookie in enumerate(row)] for row in cookies]
        marked.clear()

    # Main loop
    while True:
        # Mark cookies in rows
        for row in range(H):
            mark_row(row)

        # Mark cookies in columns
        for col in range(W):
            mark_col(col)

        # If no cookies are marked, break the loop
        if not marked:
            break

        # Remove marked cookies
        remove_marked()

    # Count the remaining cookies
    remaining = sum(1 for row in cookies for cookie in row if cookie != '.')

    return remaining

# Print the result
print(count_remaining_cookies())