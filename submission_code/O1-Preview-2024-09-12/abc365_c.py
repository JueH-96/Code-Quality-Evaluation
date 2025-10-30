import sys
import threading
def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    total_A = sum(A)
    if total_A <= M:
        print("infinite")
        return
    A.sort()
    left, right = 0, max(A)
    while left < right:
        mid = (left + right + 1) // 2
        total = 0
        for a in A:
            total += min(mid, a)
        if total <= M:
            left = mid
        else:
            right = mid -1
    print(left)
threading.Thread(target=main).start()