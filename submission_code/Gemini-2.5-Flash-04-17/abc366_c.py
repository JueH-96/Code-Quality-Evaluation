import sys

def solve():
    # Read the number of queries
    Q = int(sys.stdin.readline())

    # Use a dictionary to store counts of each ball type
    # Key: integer on the ball, Value: number of balls with that integer
    counts = {}
    # Use a variable to track the number of different (distinct) integer values
    # currently written on the balls in the bag.
    distinct_count = 0

    # Process each query
    for _ in range(Q):
        # Read the query line and parse it into a list of integers.
        # The first element is the query type, the second (if present) is the value x.
        query = list(map(int, sys.stdin.readline().split()))

        query_type = query[0]

        if query_type == 1:
            # Type 1: Put one ball with integer x written on it into the bag.
            x = query[1]
            # Get the current count of balls with integer x. If x is not yet in counts,
            # counts.get(x, 0) returns 0.
            current_count = counts.get(x, 0)

            # Increment the count for x.
            counts[x] = current_count + 1

            # If the count of x was 0 before adding (meaning it just became 1),
            # it signifies that a new distinct integer value is now present in the bag.
            # We increment the distinct_count.
            if current_count == 0:
                distinct_count += 1

        elif query_type == 2:
            # Type 2: Remove one ball with integer x written on it from the bag.
            # It is guaranteed that the bag has a ball with the integer x written on it
            # when this query is given.
            x = query[1]

            # Decrement the count for x.
            # Since we are guaranteed a ball with x exists, counts[x] must be > 0.
            counts[x] -= 1

            # If the count of x becomes 0 after removing, it means we just removed
            # the last ball with the integer x. Now, the integer x is no longer
            # represented in the bag. We decrement the distinct_count.
            if counts[x] == 0:
                distinct_count -= 1

        elif query_type == 3:
            # Type 3: Print the number of different integers written on the balls in the bag.
            # The distinct_count variable holds exactly this value.
            print(distinct_count)

# Execute the solve function to run the program logic.
solve()