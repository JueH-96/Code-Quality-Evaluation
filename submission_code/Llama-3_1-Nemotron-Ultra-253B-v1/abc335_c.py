import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    prefix = [(0, 0, 0, 0)]  # (R, L, U, D)
    
    for _ in range(Q):
        query = input[ptr]
        ptr += 1
        if query == '1':
            C = input[ptr]
            ptr += 1
            r, l, u, d = prefix[-1]
            if C == 'R':
                r += 1
            elif C == 'L':
                l += 1
            elif C == 'U':
                u += 1
            elif C == 'D':
                d += 1
            prefix.append((r, l, u, d))
        else:
            p = int(input[ptr])
            ptr += 1
            K = len(prefix) - 1
            if K >= p - 1:
                m = K - (p - 1)
                r, l, u, d = prefix[m]
                x = 1 + (r - l)
                y = u - d
            else:
                x = p - K
                y = 0
            print(x, y)

if __name__ == "__main__":
    main()