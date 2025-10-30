def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    M, D = map(int, data[:2])
    y, m, d = map(int, data[2:5])
    
    # Add one day
    d += 1
    if d > D:
        d = 1
        m += 1
        if m > M:
            m = 1
            y += 1
    
    print(y, m, d)

def main():
    solve()

if __name__ == "__main__":
    main()