def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    
    s = str(N)
    L = len(s)
    if L <= 3:
        # For N in [0, 999], print as is
        print(N)
    else:
        # Zero out the last (L - 3) digits
        d = L - 3
        base = 10 ** d
        approx = (N // base) * base
        print(approx)

if __name__ == "__main__":
    main()