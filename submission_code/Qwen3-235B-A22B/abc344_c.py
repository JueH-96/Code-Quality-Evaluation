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
    X = data[ptr:ptr+Q]
    
    # Precompute all possible a + b
    sum_ab = {a + b for a in A for b in B}
    
    # Precompute all possible (a + b) + c
    possible_sums = {ab + c for ab in sum_ab for c in C}
    
    # Prepare results
    results = []
    for x in X:
        if x in possible_sums:
            results.append("Yes")
        else:
            results.append("No")
    
    print('
'.join(results))

if __name__ == "__main__":
    main()