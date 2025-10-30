# YOUR CODE HERE
import sys
import threading
def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        a = list(map(int, sys.stdin.readline().split()))
        res = a[0]
        dp = a[0]
        for i in range(1, n):
            if (a[i] % 2) != (a[i -1] % 2):
                dp = max(a[i], dp + a[i])
            else:
                dp = a[i]
            res = max(res, dp)
        print(res)
threading.Thread(target=main).start()