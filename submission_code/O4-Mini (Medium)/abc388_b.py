def main():
    import sys
    data = sys.stdin.read().split()
    N, D = map(int, data[:2])
    entries = list(map(int, data[2:]))
    T = entries[0::2]
    L = entries[1::2]
    
    out = []
    for k in range(1, D+1):
        max_weight = 0
        for i in range(N):
            weight = T[i] * (L[i] + k)
            if weight > max_weight:
                max_weight = weight
        out.append(str(max_weight))
    
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()