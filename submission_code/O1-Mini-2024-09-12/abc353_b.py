# YOUR CODE HERE
def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    from collections import deque
    queue = deque(A)
    empty_seats = K
    count_starts = 0
    while True:
        if not queue:
            count_starts +=1
            break
        front = queue[0]
        if empty_seats < front:
            count_starts +=1
            empty_seats = K
        else:
            queue.popleft()
            empty_seats -= front
    print(count_starts)

if __name__ == "__main__":
    main()