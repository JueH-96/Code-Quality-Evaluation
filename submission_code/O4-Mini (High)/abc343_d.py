import sys

def main():
    import sys
    input = sys.stdin.readline

    N, T = map(int, input().split())
    scores = [0] * N
    freq = {0: N}
    distinct = 1

    out = []
    for _ in range(T):
        a, b = map(int, input().split())
        idx = a - 1

        old = scores[idx]
        new = old + b

        # remove one occurrence of the old score
        cnt_old = freq[old]
        if cnt_old == 1:
            del freq[old]
            distinct -= 1
        else:
            freq[old] = cnt_old - 1

        # add one occurrence of the new score
        cnt_new = freq.get(new, 0)
        if cnt_new == 0:
            freq[new] = 1
            distinct += 1
        else:
            freq[new] = cnt_new + 1

        scores[idx] = new
        out.append(str(distinct))

    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()