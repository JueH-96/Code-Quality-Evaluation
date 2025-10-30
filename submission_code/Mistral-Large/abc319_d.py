import sys

def can_fit_in_m_lines(width, n, m, lengths):
    current_line_length = 0
    current_line_count = 1

    for i in range(n):
        if current_line_length + lengths[i] <= width:
            current_line_length += lengths[i] + 1
        else:
            if current_line_count == m:
                return False
            current_line_length = lengths[i] + 1
            current_line_count += 1

    return True

def find_minimum_width(n, m, lengths):
    left, right = max(lengths), sum(lengths) + n - 1
    while left < right:
        mid = (left + right) // 2
        if can_fit_in_m_lines(mid, n, m, lengths):
            right = mid
        else:
            left = mid + 1
    return left

def main():
    input = sys.stdin.read
    data = input().split()

    n = int(data[0])
    m = int(data[1])
    lengths = list(map(int, data[2:]))

    result = find_minimum_width(n, m, lengths)
    print(result)

if __name__ == "__main__":
    main()