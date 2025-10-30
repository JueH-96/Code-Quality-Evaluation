# YOUR CODE HERE
import sys

def main():
    """
    Solves the AtCoder City garbage collection problem by processing Q queries
    based on N garbage collection schedules.
    """
    # Use a faster method for reading input lines, common in competitive programming.
    input = sys.stdin.readline

    # Read N, the number of garbage types.
    try:
        N = int(input())
    except (ValueError, IndexError):
        # This handles cases like empty input at the beginning.
        return

    # Read the N collection schedules (q_i, r_i) using a list comprehension.
    # schedules[i] will store the schedule for garbage type i+1.
    schedules = [tuple(map(int, input().split())) for _ in range(N)]
    
    # Read Q, the number of queries.
    try:
        Q = int(input())
    except (ValueError, IndexError):
        return

    # Process each of the Q queries.
    for _ in range(Q):
        try:
            t, d = map(int, input().split())
        except (ValueError, IndexError):
            # Stop if query input is malformed or incomplete.
            break
            
        # Retrieve the schedule (q, r) for the given garbage type t.
        # The problem uses 1-based indexing for t, so we use t-1 for 0-indexed list access.
        q, r = schedules[t - 1]
        
        # Calculate the remainder of the given day d modulo q. Let's call it s.
        s = d % q
        
        # Calculate the number of days 'diff' to add to d to reach the next collection day.
        # The formula `(r - s + q) % q` correctly finds the smallest non-negative
        # offset, handling cases where d is before, on, or after the collection
        # day within its q-day cycle. The `+ q` ensures the first operand to '%' is non-negative.
        diff = (r - s + q) % q
        
        # The next collection day is the current day d plus the calculated difference.
        answer = d + diff
        
        # Print the result for the current query to standard output.
        print(answer)

if __name__ == "__main__":
    main()