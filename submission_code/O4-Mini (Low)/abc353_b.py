def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    empty = K
    count = 0
    i = 0

    while True:
        # If queue is empty, start the ride one last time and finish.
        if i >= N:
            count += 1
            break

        # If the next group doesn't fit, start the ride now.
        if A[i] > empty:
            count += 1
            empty = K
        else:
            # Otherwise, seat the group and remove them from the queue.
            empty -= A[i]
            i += 1

    print(count)


if __name__ == "__main__":
    main()