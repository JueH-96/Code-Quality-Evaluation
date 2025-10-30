import collections

def solve():
    # Read K from standard input
    K = int(input())

    # Initialize a deque (double-ended queue) for BFS.
    # This queue will store 321-like numbers in increasing order as they are discovered.
    q = collections.deque()

    # Start with all single-digit 321-like numbers (1 through 9).
    # These are the base cases for generating longer numbers.
    # Appending them in ascending order helps maintain the sorted property of the queue initially.
    for i in range(1, 10):
        q.append(i)

    # Initialize a counter for the found 321-like numbers.
    count = 0
    # Variable to store the K-th smallest number.
    result = -1

    # Perform BFS until the K-th number is found
    while q:
        # Get the smallest available 321-like number from the front of the queue
        current_num = q.popleft()
        count += 1

        # If this is the K-th number we are looking for, store it and break the loop
        if count == K:
            result = current_num
            break

        # Get the last digit of the current number.
        # New digits must be strictly smaller than this last digit to maintain the 321-like property.
        last_digit = current_num % 10

        # Iterate through possible next digits (from 0 up to last_digit - 1).
        # We iterate in increasing order (0, 1, 2, ..., last_digit-1) to ensure
        # that new numbers generated are added to the queue in increasing order,
        # preserving the overall sorted property of the BFS.
        for next_digit in range(0, last_digit):
            # Form the new number by appending next_digit
            new_num = current_num * 10 + next_digit
            # Add the new 321-like number to the back of the queue
            q.append(new_num)
            
    # Print the K-th smallest 321-like number
    print(result)

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()