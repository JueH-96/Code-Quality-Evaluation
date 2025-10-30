import sys
import threading

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    A = data[1:]
    # succ[i] will store the person standing right behind person i
    succ = [0] * (n + 1)
    head = 0
    for i in range(1, n + 1):
        ai = int(A[i-1])
        if ai == -1:
            head = i
        else:
            succ[ai] = i

    # Reconstruct the line by following succ[] from the head
    res = []
    cur = head
    for _ in range(n):
        res.append(str(cur))
        cur = succ[cur]

    sys.stdout.write(" ".join(res))

if __name__ == "__main__":
    main()