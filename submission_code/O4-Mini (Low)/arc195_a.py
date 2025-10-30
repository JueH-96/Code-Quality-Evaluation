import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    m = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(m)]
    # Find earliest match positions L
    L = [0] * m
    j = 0
    for i in range(n):
        if j < m and A[i] == B[j]:
            L[j] = i
            j += 1
            if j == m:
                break
    if j < m:
        # can't even match once
        print("No")
        return
    # Find latest match positions R
    R = [0] * m
    j = m - 1
    for i in range(n-1, -1, -1):
        if j >= 0 and A[i] == B[j]:
            R[j] = i
            j -= 1
            if j < 0:
                break
    # Check if there's any position where we have two choices
    for k in range(m):
        if R[k] > L[k]:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()