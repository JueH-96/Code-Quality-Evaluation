# YOUR CODE HERE

import sys

import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    N, S = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    total_sum = sum(A)
    prefix_sum = [0]*(N+1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i]+A[i]
    max_window_size = 2*N  # Adjusted, as longer windows repeat sums

    cumsum = [0]
    seen_sums = set()
    seen_sums.add(0)
    found = False
    for i in range(1, max_window_size+1):
        cumsum.append(cumsum[i-1]+A[(i-1)%N])
        if cumsum[i] - S in seen_sums:
            print('Yes')
            found = True
            break
        seen_sums.add(cumsum[i])
    if not found:
        print('No')

if __name__ == '__main__':
    threading.Thread(target=main).start()