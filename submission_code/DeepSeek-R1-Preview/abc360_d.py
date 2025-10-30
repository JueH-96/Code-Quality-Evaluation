import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    T = int(input[idx])
    idx += 1
    S = input[idx]
    idx += 1
    X = list(map(int, input[idx:idx+N]))
    
    left = []
    right = []
    for s, x in zip(S, X):
        if s == '0':
            left.append(x)
        else:
            right.append(x)
    
    left_sorted = sorted(left)
    total = 0
    
    for x in right:
        a = x + 1
        b = x + 2 * T
        l = bisect.bisect_left(left_sorted, a)
        r = bisect.bisect_right(left_sorted, b)
        total += r - l
    
    print(total)

if __name__ == '__main__':
    main()