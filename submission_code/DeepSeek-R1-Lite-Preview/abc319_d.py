import sys

def main():
    # Read all input
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    L = list(map(int, data[2:2+N]))
    
    # Determine left and right for binary search
    left = max(L)
    right = sum(L) + (N - 1)
    
    # Binary search
    while left < right:
        mid = (left + right) // 2
        if can_fit(mid, L, M):
            right = mid
        else:
            left = mid + 1
    print(left)

def can_fit(W, L, M):
    lines_used = 1
    current_width = 0
    for word_width in L:
        if current_width == 0:
            # First word in the line
            current_width = word_width
        else:
            # Need to add a space
            if current_width + 1 + word_width <= W:
                current_width += 1 + word_width
            else:
                # Start a new line
                lines_used += 1
                current_width = word_width
        if lines_used > M:
            return False
    return True

if __name__ == '__main__':
    main()