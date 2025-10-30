import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Read N (number of words) and M (max lines) from standard input.
    # Using sys.stdin.readline for efficiency with large inputs.
    try:
        line1 = sys.stdin.readline()
        if not line1: return
        N, M = map(int, line1.split())
        
        line2 = sys.stdin.readline()
        if not line2: return
        L = list(map(int, line2.split()))
    except (IOError, ValueError):
        # Gracefully handle potential empty lines or malformed input.
        return

    def check(W):
        """
        Greedily checks if the words can be placed in at most M lines
        given a maximum window width of W.
        Returns True if possible, False otherwise.
        """
        lines_used = 1
        current_line_width = 0
        
        for word_width in L:
            # The lower bound of the binary search ensures W >= max(L),
            # so any single word can fit on a line by itself.
            
            # If the current line is empty, place the first word.
            if current_line_width == 0:
                current_line_width = word_width
            # Else, check if the word fits on the current line (with a preceding space).
            elif current_line_width + 1 + word_width <= W:
                current_line_width += 1 + word_width
            # If the word doesn't fit, start a new line.
            else:
                lines_used += 1
                current_line_width = word_width
        
        return lines_used <= M

    # --- Binary Search for the Minimum Width ---
    
    # Lower bound: The width must be at least that of the longest single word.
    left = max(L)
    
    # Upper bound: A safe upper limit is the width for all words on one line.
    right = sum(L) + (N - 1)
    
    # 'ans' will store the minimum valid width found.
    ans = right

    while left <= right:
        # Calculate the middle of the current search range.
        mid = left + (right - left) // 2
        
        if check(mid):
            # If `mid` is a valid width, it's a potential answer.
            # We store it and try to find an even smaller valid width.
            ans = mid
            right = mid - 1
        else:
            # If `mid` is not a valid width, we need a larger one.
            left = mid + 1

    # Print the final minimum width.
    print(ans)

solve()