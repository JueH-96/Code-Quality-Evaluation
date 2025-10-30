# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    import threading

    def solve():
        N, M, D = map(int, sys.stdin.readline().split())
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
        A.sort()
        B.sort()
        max_sum = -1
        for a in A:
            index = bisect.bisect_right(B, a + D) -1
            if index >= 0 and B[index] >= a - D:
                max_sum = max(max_sum, a + B[index])
        print(max_sum)

    threading.Thread(target=solve,).start()

if __name__ == "__main__":
    main()