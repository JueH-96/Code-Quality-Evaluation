import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    counters = []
    K = []
    for _ in range(N):
        data = list(map(int, input().split()))
        k = data[0]
        arr = data[1:]
        d = {}
        for v in arr:
            d[v] = d.get(v, 0) + 1
        counters.append(d)
        K.append(k)

    max_prob = 0.0
    for i in range(N):
        di = counters[i]
        Ki = K[i]
        len_di = len(di)
        for j in range(i+1, N):
            dj = counters[j]
            Kj = K[j]
            denom = Ki * Kj

            # iterate over the smaller dictionary for speed
            if len_di <= len(dj):
                small, big = di, dj
            else:
                small, big = dj, di

            get_big = big.get
            numerator = 0
            for x, cnt_small in small.items():
                cnt_big = get_big(x)
                if cnt_big:
                    numerator += cnt_small * cnt_big

            prob = numerator / denom
            if prob > max_prob:
                max_prob = prob

    # print with sufficient precision
    print("{:.12f}".format(max_prob))

if __name__ == "__main__":
    main()