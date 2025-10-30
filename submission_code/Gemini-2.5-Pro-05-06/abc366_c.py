import sys

def main():
    # Q: Number of queries
    Q = int(sys.stdin.readline())

    # counts_arr[x] stores the count of balls with integer x.
    # Max x is 10^6, so list size is 10^6 + 1.
    # Python lists are 0-indexed. counts_arr[0] will be unused as x >= 1.
    counts_arr = [0] * (1000001) 
    
    # distinct_count stores the number of unique integers currently in the bag.
    distinct_count = 0

    for _ in range(Q):
        query_parts = sys.stdin.readline().split()
        query_type = int(query_parts[0])

        if query_type == 1:
            x = int(query_parts[1])
            # If x was not in the bag (its count was 0), 
            # adding it makes it a new distinct number.
            if counts_arr[x] == 0:
                distinct_count += 1
            # Increment the count of balls with integer x.
            counts_arr[x] += 1
        elif query_type == 2:
            x = int(query_parts[1])
            # It's guaranteed that a ball with x is in the bag, so counts_arr[x] > 0.
            # Decrement the count of balls with integer x.
            counts_arr[x] -= 1
            # If the count of x becomes 0, it's no longer a distinct number in the bag.
            if counts_arr[x] == 0:
                distinct_count -= 1
        elif query_type == 3:
            # Print the current number of distinct integers.
            print(distinct_count)

if __name__ == '__main__':
    main()