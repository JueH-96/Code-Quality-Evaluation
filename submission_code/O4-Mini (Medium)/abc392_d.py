import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    # Read dice: store each die's face-count dictionary and its number of faces
    counts = []
    sizes = []
    for _ in range(N):
        data = list(map(int, input().split()))
        k = data[0]
        faces = data[1:]
        sizes.append(k)
        cnt = {}
        for v in faces:
            cnt[v] = cnt.get(v, 0) + 1
        counts.append(cnt)

    best = 0.0
    # Check every pair of dice
    for i in range(N):
        cnt_i = counts[i]
        ki = sizes[i]
        for j in range(i + 1, N):
            cnt_j = counts[j]
            kj = sizes[j]
            # iterate over the smaller dictionary for efficiency
            if len(cnt_i) < len(cnt_j):
                small, big = cnt_i, cnt_j
            else:
                small, big = cnt_j, cnt_i

            # compute sum of cnt_i[v] * cnt_j[v] over common v
            common = 0
            for v, ci in small.items():
                cj = big.get(v)
                if cj is not None:
                    common += ci * cj

            prob = common / (ki * kj)
            if prob > best:
                best = prob

    # print with sufficient precision
    print("{:.15f}".format(best))


if __name__ == "__main__":
    main()