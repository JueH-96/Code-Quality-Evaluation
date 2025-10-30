# YOUR CODE HERE

import sys

import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)

    N, M = map(int, sys.stdin.readline().split())
    L = list(map(int, sys.stdin.readline().split()))
    
    max_Li = max(L)
    total_length = sum(L) + N -1  # Max possible width

    left = max_Li  # minimal possible width
    right = total_length  # maximal possible width

    while left < right:
        mid = (left + right)//2
        lines_needed = 1
        current_line_width = L[0]
        for i in range(1, N):
            if current_line_width + 1 + L[i] <= mid:
                current_line_width += 1 + L[i]
            else:
                lines_needed +=1
                current_line_width = L[i]

        if lines_needed <= M:
            right = mid
        else:
            left = mid + 1
    print(left)


if __name__ == "__main__":
    threading.Thread(target=main).start()