# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    min_prefix = 0
    prefix_sum = 0
    total_sum = 0
    for a in A:
        prefix_sum += a
        min_prefix = min(min_prefix, prefix_sum)
        total_sum += a
    x = max(0, -min_prefix)
    print(x + total_sum)

threading.Thread(target=main).start()