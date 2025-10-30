# YOUR CODE HERE
import sys
import threading
def main():
    import bisect
    import sys
    import threading
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    A = list(map(int, input_data[1:n+1]))
    M = 10**8
    n = len(A)
    A.sort()
    prefix_sum = [0] * (n+1)
    for i in range(n):
        prefix_sum[i+1] = prefix_sum[i] + A[i]
    total_sum_unmod = 0
    for i in range(n-1):
        count = n - i -1
        sum_rest = prefix_sum[n] - prefix_sum[i+1]
        total_sum_unmod += A[i] * count + sum_rest
    overflows = 0
    for i in range(n):
        idx = bisect.bisect_left(A, M - A[i], i+1)
        overflows += n - idx
    total_sum = total_sum_unmod - overflows * M
    print(total_sum)
threading.Thread(target=main).start()