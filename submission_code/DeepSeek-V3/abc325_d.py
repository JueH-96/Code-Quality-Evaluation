# YOUR CODE HERE
import sys
import heapq

def main():
    N = int(sys.stdin.readline())
    products = []
    for _ in range(N):
        T, D = map(int, sys.stdin.readline().split())
        products.append((T, T + D))
    
    # Sort by end time
    products.sort(key=lambda x: x[1])
    
    max_prints = 0
    last_print_time = -1
    
    for start, end in products:
        # The earliest time we can print is max(start, last_print_time + 1)
        earliest_print = max(start, last_print_time + 1)
        if earliest_print <= end:
            max_prints += 1
            last_print_time = earliest_print
    
    print(max_prints)

if __name__ == "__main__":
    main()