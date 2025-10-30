import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    garbage = []
    for _ in range(N):
        q = int(input[ptr])
        r = int(input[ptr+1])
        garbage.append((q, r))
        ptr += 2
    Q = int(input[ptr])
    ptr += 1
    for _ in range(Q):
        t = int(input[ptr])
        d = int(input[ptr+1])
        ptr += 2
        q, r = garbage[t-1]
        numerator = (d - r) + (q - 1)
        m = numerator // q
        x = r + m * q
        print(x)

if __name__ == "__main__":
    main()