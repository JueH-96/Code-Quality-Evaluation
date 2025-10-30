def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    L = list(map(int, data[2:]))

    # Function to check if the sentence can fit into M or fewer lines with width W
    def can_fit(W):
        lines_used = 1
        current_line_width = L[0]
        
        for i in range(1, N):
            # Check if we can place the next word in the same line
            if current_line_width + 1 + L[i] <= W:
                current_line_width += 1 + L[i]
            else:
                # Otherwise, move to the next line
                lines_used += 1
                current_line_width = L[i]
                # If we exceed M lines, it doesn't fit
                if lines_used > M:
                    return False
        return True

    # The minimum possible width is the max length of a single word
    left = max(L)
    # The maximum possible width is if all words are on one line:
    # sum of lengths plus (N-1) spaces
    right = sum(L) + (N - 1)

    # Binary search for the minimum width
    while left < right:
        mid = (left + right) // 2
        if can_fit(mid):
            right = mid
        else:
            left = mid + 1

    print(left)

def main():
    solve()

if __name__ == "__main__":
    main()