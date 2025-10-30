import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    M = 1000000
    
    freq = [0] * (M + 1)
    for a in A:
        if a <= M:
            freq[a] += 1

    G = [0] * (M + 1)
    for d in range(1, M + 1):
        j = d
        while j <= M:
            G[d] += freq[j]
            j += d

    candidate = [0] * (M + 1)
    for d in range(M, 0, -1):
        if G[d] >= k:
            j = d
            while j <= M:
                if candidate[j] == 0:
                    candidate[j] = d
                j += d

    out_lines = [str(candidate[a]) for a in A]
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()