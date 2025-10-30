def main():
    import sys
    data = sys.stdin.read().split()
    N, T, P = map(int, data[:3])
    L = list(map(int, data[3:3+N]))
    # For each person, compute how many days until hair >= T (0 if already >= T)
    deficits = [max(0, T - li) for li in L]
    deficits.sort()
    # The P-th smallest deficit is the answer (0-based index P-1)
    print(deficits[P-1])

if __name__ == "__main__":
    main()