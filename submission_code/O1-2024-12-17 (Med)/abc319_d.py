def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    L = list(map(int, data[2:]))

    # The minimum width needed is at least the maximum single word length
    left = max(L)
    # The maximum width (if all words go on one line) is sum of all word lengths + spaces between them
    right = sum(L) + (N - 1)

    # Function to check if we can fit all words in M lines with window width mid
    def can_fit(mid):
        line_count = 1  # number of lines used
        current_width = L[0]  # width usage of the current line

        for i in range(1, N):
            # If this word can be placed in the same line with a space
            if current_width + 1 + L[i] <= mid:
                current_width += 1 + L[i]
            else:
                # Move to the next line
                line_count += 1
                current_width = L[i]
                if line_count > M:  # If we exceed M lines, return False
                    return False
        return True

    # Binary search for the minimum window width
    answer = right
    while left <= right:
        mid = (left + right) // 2
        if can_fit(mid):
            answer = mid
            right = mid - 1
        else:
            left = mid + 1

    print(answer)

# Do not forget to call main()!
main()