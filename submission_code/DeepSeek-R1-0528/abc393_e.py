import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    M = 1000000

    freq = [0] * (M+1)
    for a in A:
        if a <= M:
            freq[a] += 1

    count_arr = [0] * (M+1)
    for d in range(1, M+1):
        total = 0
        j = d
        while j <= M:
            total += freq[j]
            j += d
        count_arr[d] = total

    best = [0] * (M+1)
    for d in range(M, 0, -1):
        if count_arr[d] < k:
            continue
        j = d
        while j <= M:
            if best[j] == 0:
                best[j] = d
            j += d

    for a in A:
        print(best[a])

if __name__ == "__main__":
    main()