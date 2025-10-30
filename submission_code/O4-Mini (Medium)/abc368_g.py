import threading
import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    Q = int(input())
    out = []
    for _ in range(Q):
        q = input().split()
        t = int(q[0])
        if t == 1:
            i = int(q[1]) - 1
            x = int(q[2])
            A[i] = x
        elif t == 2:
            i = int(q[1]) - 1
            x = int(q[2])
            B[i] = x
        else:  # t == 3
            l = int(q[1]) - 1
            r = int(q[2]) - 1
            v = 0
            # DP: at each i, v = max(v + A[i], v * B[i])
            for i in range(l, r + 1):
                # compute both options
                add = v + A[i]
                mul = v * B[i]
                # choose the best
                if mul > add:
                    v = mul
                else:
                    v = add
            out.append(str(v))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()