def main():
    import sys
    n, T, P = map(int, sys.stdin.readline().split())
    L = list(map(int, sys.stdin.readline().split()))
    
    base = 0
    deficits = []
    for x in L:
        if x >= T:
            base += 1
        else:
            deficits.append(T - x)
            
    if base >= P:
        print(0)
    else:
        deficits.sort()
        k = P - base
        print(deficits[k-1])

if __name__ == "__main__":
    main()