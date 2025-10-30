import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    # store for every die:
    #  - total faces K_i
    #  - dictionary value -> count on this die
    Ks = []
    counts = []   # list of dicts
    for _ in range(N):
        K = int(next(it))
        Ks.append(K)
        d = {}
        for _ in range(K):
            v = int(next(it))
            d[v] = d.get(v, 0) + 1
        counts.append(d)
    
    max_prob = 0.0
    
    for i in range(N):
        Ki = Ks[i]
        di = counts[i]
        for j in range(i + 1, N):
            Kj = Ks[j]
            dj = counts[j]
            # iterate over smaller dictionary for efficiency
            if len(di) <= len(dj):
                small, other = di, dj
            else:
                small, other = dj, di
            dot = 0
            for v, cnt in small.items():
                if v in other:
                    dot += cnt * other[v]
            prob = dot / (Ki * Kj)
            if prob > max_prob:
                max_prob = prob
    
    # print with enough precision
    print("{:.15f}".format(max_prob))

if __name__ == "__main__":
    main()