import sys
import threading

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    last = [0] * (n + 1)
    res = 0
    # For each occurrence at position i of value v, let prev be the
    # previous occurrence of v (0 if none). Then the number of
    # subarrays where this occurrence is the first appearance of v
    # is (i-prev) * (n-i+1).
    for i, v in enumerate(A, start=1):
        prev = last[v]
        res += (i - prev) * (n - i + 1)
        last[v] = i
    print(res)

if __name__ == "__main__":
    main()