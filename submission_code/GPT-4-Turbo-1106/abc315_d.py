def remove_marked_cookies(cookies, H, W):
    while True:
        marked_rows = set()
        marked_cols = set()

        # Mark rows with all same color cookies
        for i in range(H):
            if len(set(cookies[i])) == 1 and cookies[i][0] != '.':
                marked_rows.add(i)

        # Mark columns with all same color cookies
        for j in range(W):
            column = [cookies[i][j] for i in range(H)]
            if len(set(column)) == 1 and column[0] != '.':
                marked_cols.add(j)

        # If no cookies are marked, break the loop
        if not marked_rows and not marked_cols:
            break

        # Remove marked cookies
        for i in marked_rows:
            cookies[i] = ['.'] * W
        for j in marked_cols:
            for i in range(H):
                cookies[i][j] = '.'

    # Count remaining cookies
    remaining_cookies = sum(row.count('.') == 0 for row in cookies)
    return remaining_cookies

def main():
    H, W = map(int, input().split())
    cookies = [list(input().strip()) for _ in range(H)]
    remaining_cookies = remove_marked_cookies(cookies, H, W)
    print(remaining_cookies)

if __name__ == "__main__":
    main()