import sys
from collections import deque

def main():
    """
    This function encapsulates the logic to solve the problem.
    It reads from standard input and writes to standard output.
    """
    # Use fast I/O
    input = sys.stdin.readline
    
    # Read the number of queries
    try:
        Q = int(input())
    except (ValueError, IndexError):
        # Handle empty input case
        return

    # Deque to store groups of plants as (base_height, count) tuples.
    # The base_heights in the deque will be in descending order.
    plants = deque()
    
    # global_offset tracks the total growth from all type 2 queries.
    global_offset = 0

    for _ in range(Q):
        query = input().split()
        q_type = int(query[0])

        if q_type == 1:
            # Plant a new flower. Its initial height is 0.
            # Its base_height must satisfy: base_height + global_offset = 0
            base_height = -global_offset
            
            # If the most recently added group has the same base_height,
            # increment its count. This groups consecutive type 1 queries.
            if plants and plants[-1][0] == base_height:
                # Tuples are immutable, so pop the last element and append an updated one.
                _, count = plants.pop()
                plants.append((base_height, count + 1))
            else:
                # Otherwise, start a new group.
                plants.append((base_height, 1))

        elif q_type == 2:
            # Wait for T days. All plants grow by T.
            T = int(query[1])
            global_offset += T

        elif q_type == 3:
            # Harvest all plants with height at least H.
            H = int(query[1])
            # The harvest condition is base_height + global_offset >= H,
            # which is equivalent to base_height >= H - global_offset.
            threshold = H - global_offset
            
            harvested_count = 0
            # Check from the left of the deque, which has plants with the largest base_height.
            while plants and plants[0][0] >= threshold:
                # This group of plants must be harvested.
                _, count = plants.popleft()
                harvested_count += count
            
            # Print the number of harvested plants.
            print(harvested_count)

if __name__ == "__main__":
    main()