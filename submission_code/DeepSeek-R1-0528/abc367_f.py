import random
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    
    rand0 = [0] * (N + 1)
    rand1 = [0] * (N + 1)
    for i in range(1, N + 1):
        rand0[i] = random.getrandbits(64)
        rand1[i] = random.getrandbits(64)
    
    prefixA0 = [0] * (N + 1)
    prefixA1 = [0] * (N + 1)
    for i in range(1, N + 1):
        num = A[i - 1]
        prefixA0[i] = prefixA0[i - 1] + rand0[num]
        prefixA1[i] = prefixA1[i - 1] + rand1[num]
    
    prefixB0 = [0] * (N + 1)
    prefixB1 = [0] * (N + 1)
    for i in range(1, N + 1):
        num = B[i - 1]
        prefixB0[i] = prefixB0[i - 1] + rand0[num]
        prefixB1[i] = prefixB1[i - 1] + rand1[num]
    
    out_lines = []
    for _ in range(Q):
        l = int(next(it))
        r = int(next(it))
        L = int(next(it))
        R = int(next(it))
        lenA = r - l + 1
        lenB = R - L + 1
        if lenA != lenB:
            out_lines.append("No")
        else:
            a0 = prefixA0[r] - prefixA0[l - 1]
            a1 = prefixA1[r] - prefixA1[l - 1]
            b0 = prefixB0[R] - prefixB0[L - 1]
            b1 = prefixB1[R] - prefixB1[L - 1]
            if a0 == b0 and a1 == b1:
                out_lines.append("Yes")
            else:
                out_lines.append("No")
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()