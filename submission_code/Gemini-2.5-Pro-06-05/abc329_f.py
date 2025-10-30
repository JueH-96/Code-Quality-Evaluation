# YOUR CODE HERE
import sys

def main():
    """
    Reads input from stdin, solves the problem, and writes the answer to stdout.
    """
    # Use a faster method for reading input, crucial for large N and Q
    input = sys.stdin.readline

    # Read the number of boxes (N) and queries (Q)
    try:
        N, Q = map(int, input().split())
    except (IOError, ValueError):
        # Gracefully exit if input is empty or malformed
        return

    # Read the initial colors for each box
    C = list(map(int, input().split()))

    # --- Data Structures for Efficient Merging ---

    # `box_location[i]` stores the index (pointer) to the container holding the
    # balls for box `i`. We use 1-based indexing for boxes.
    # A pointer of 0 indicates an empty box.
    box_location = list(range(N + 1))

    # `containers[k]` is the set of distinct colors for the group of balls `k`.
    # `containers[0]` is an empty set, representing the state of an empty box.
    # Initially, box `i` has its own container `i` with one ball of color C[i-1].
    containers = [set()] + [{C[i - 1]} for i in range(1, N + 1)]

    # --- Process Each Query ---
    for _ in range(Q):
        try:
            a, b = map(int, input().split())
        except (IOError, ValueError):
            # Stop if there are no more queries to read
            break

        # Get the pointers to the containers for boxes `a` and `b`
        a_ptr = box_location[a]
        b_ptr = box_location[b]

        # If box `a`'s balls are already in the same group as box `b`'s
        # (which includes the case where both are empty), then `a` is effectively
        # empty with respect to `b`. Nothing needs to be moved.
        if a_ptr == b_ptr:
            print(len(containers[b_ptr]))
            continue

        # To merge efficiently, always iterate over the smaller set and add its
        # elements to the larger set. This is the "merge smaller into larger" heuristic.
        
        # Case 1: The container for `a` has more distinct colors than `b`'s.
        if len(containers[a_ptr]) > len(containers[b_ptr]):
            # Merge the colors from `b`'s container into `a`'s.
            containers[a_ptr].update(containers[b_ptr])
            
            # The merged collection is now considered to be in box `b`.
            # So, box `b` now points to `a`'s original container.
            box_location[b] = a_ptr
            
            # Box `a` is now empty.
            box_location[a] = 0
            
            # The container `b` originally pointed to is now unused. Clear it to free memory.
            containers[b_ptr] = set()
            
            # The result is the number of distinct colors in the new container of box `b`.
            print(len(containers[a_ptr]))
        
        # Case 2: The container for `b` has more or an equal number of colors.
        else:
            # Merge the colors from `a`'s container into `b`'s.
            containers[b_ptr].update(containers[a_ptr])
            
            # Box `b` keeps its pointer, as it now holds the merged collection.
            # Box `a` is now empty.
            box_location[a] = 0
            
            # The container `a` originally pointed to is now unused.
            containers[a_ptr] = set()
            
            # The result is the number of distinct colors in box `b`'s container.
            print(len(containers[b_ptr]))

if __name__ == "__main__":
    main()