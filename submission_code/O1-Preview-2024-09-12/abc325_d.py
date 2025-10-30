# YOUR CODE HERE
import sys
import threading

def main():
    N = int(sys.stdin.readline())
    intervals = []
    for _ in range(N):
        T_i, D_i = map(int, sys.stdin.readline().split())
        start_i = T_i
        end_i = T_i + D_i
        intervals.append((end_i, start_i))

    intervals.sort()
    last_print_time = -1 << 60  # A very negative number
    count = 0
    for end_i, start_i in intervals:
        earliest = max(start_i, last_print_time + 1)
        if earliest <= end_i:
            count += 1
            last_print_time = earliest

    print(count)

threading.Thread(target=main).start()