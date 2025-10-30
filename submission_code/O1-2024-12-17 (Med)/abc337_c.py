def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # We will track for each person who is immediately behind them.
    # Initialize an array of size N+1, because people are 1-indexed.  
    # next_person[i] = j means person j is directly behind person i.
    next_person = [0] * (N + 1)
    
    # Find the front of the line (the person with A_i = -1)
    front = -1
    for i in range(1, N+1):
        if A[i-1] == -1:
            front = i
        else:
            # A[i-1] tells us that person i is behind person A[i-1].
            next_person[A[i-1]] = i

    # Traverse from front to the end of the line
    order = []
    current = front
    while current != 0:
        order.append(current)
        current = next_person[current]

    print(" ".join(map(str, order)))

if __name__ == "__main__":
    main()