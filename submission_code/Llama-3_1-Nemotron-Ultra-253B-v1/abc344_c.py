import sys

def main():
    data = list(map(int, sys.stdin.read().split()))
    ptr = 0
    N = data[ptr]
    ptr += 1
    A = data[ptr:ptr+N]
    ptr += N
    M = data[ptr]
    ptr += 1
    B = data[ptr:ptr+M]
    ptr += M
    L = data[ptr]
    ptr += 1
    C = data[ptr:ptr+L]
    ptr += L
    Q = data[ptr]
    ptr += 1
    Xs = data[ptr:ptr+Q]
    ptr += Q
    
    sums = set()
    for a in A:
        for b in B:
            for c in C:
                sums.add(a + b + c)
    
    for x in Xs:
        print("Yes" if x in sums else "No")

if __name__ == "__main__":
    main()