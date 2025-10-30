# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    C = list(map(int, sys.stdin.readline().split()))
    boxes = [set() for _ in range(N+1)]  # boxes[1..N]
    for i, c in enumerate(C,1):  # boxes[1..N], c in C[0..N-1]
        boxes[i].add(c)

    for _ in range(Q):
        a_str, b_str = sys.stdin.readline().split()
        a = int(a_str)
        b = int(b_str)
        if a != b:
            if len(boxes[a]) > len(boxes[b]):
                boxes[a], boxes[b] = boxes[b], boxes[a]
            boxes[b] |= boxes[a]
            boxes[a] = set()
            print(len(boxes[b]))
        else:
            # As per the constraints, a != b, so this should not happen
            pass

if __name__ == "__main__":
    threading.Thread(target=main).start()