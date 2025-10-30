from typing import List

def check(W: int, words: List[int], max_lines: int) -> bool:
    """
    Checks if it's possible to display the words in the window with a total width not exceeding W,
    fitting them into max_lines or fewer lines.
    """
    lines = 1
    line_width = 0
    for word in words:
        if line_width + word + 1 <= W:
            line_width += word + 1
        else:
            lines += 1
            line_width = word + 1
    return lines <= max_lines

def min_window_width(N: int, M: int, words: List[int]) -> int:
    """
    Finds the minimum width of the window required to display N words in M or fewer lines.
    """
    left = max(words)
    right = sum(words) + N - 1

    while left < right:
        mid = left + (right - left) // 2
        if check(mid, words, M):
            right = mid
        else:
            left = mid + 1
    return left

def main() -> None:
    """
    Main function to read inputs, process, and write the output.
    """
    N, M = map(int, input().split())
    words = list(map(int, input().split()))

    print(min_window_width(N, M, words))

if __name__ == "__main__":
    main()