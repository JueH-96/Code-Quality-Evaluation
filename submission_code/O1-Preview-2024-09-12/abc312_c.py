import sys
import bisect

def main():
    import threading
    def solve():
        N, M = map(int, sys.stdin.readline().split())
        A_list = list(map(int, sys.stdin.readline().split()))
        B_list = list(map(int, sys.stdin.readline().split()))

        A_list.sort()
        B_list.sort()

        low = 1
        high = 10**9 + 1

        while low < high:
            mid = (low + high) // 2
            S_mid = bisect.bisect_right(A_list, mid)
            B_mid = M - bisect.bisect_left(B_list, mid)

            if S_mid >= B_mid:
                high = mid
            else:
                low = mid + 1

        print(low)

    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()