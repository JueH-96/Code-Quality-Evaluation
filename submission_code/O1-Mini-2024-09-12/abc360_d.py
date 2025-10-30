import sys
import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    T = int(data[1])
    S = data[2]
    X = list(map(int, data[3:]))
    
    ants = sorted(zip(X, S), key=lambda x: x[0])
    left_positions = sorted([x for x, s in ants if s == '0'])
    total = 0
    for x, s in ants:
        if s == '1':
            upper = x + 2*T
            l = bisect.bisect_right(left_positions, upper)
            r = bisect.bisect_right(left_positions, x)
            total += l - r
    print(total)

if __name__ == "__main__":
    main()