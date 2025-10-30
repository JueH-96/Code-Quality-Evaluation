import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    
    N, M = map(int, input().split())
    L = list(map(int, input().split()))
    
    # Check if a given width W can fit the sentence into <= M lines.
    def can_fit(W):
        lines_used = 1
        curr_width = 0
        for w in L:
            if curr_width == 0:
                # Start of a new line
                curr_width = w
            else:
                # Try to place w on the same line with a preceding space
                if curr_width + 1 + w <= W:
                    curr_width += 1 + w
                else:
                    # Move to next line
                    lines_used += 1
                    if lines_used > M:
                        return False
                    curr_width = w
        return True

    # Binary search for the minimum W
    lo = max(L)              # at least the widest word
    hi = sum(L) + (N - 1)    # at most all words plus spaces in one line
    
    while lo < hi:
        mid = (lo + hi) // 2
        if can_fit(mid):
            hi = mid
        else:
            lo = mid + 1
    
    print(lo)

if __name__ == "__main__":
    main()