import sys

def main() -> None:
    input = sys.stdin.readline

    # read the first line
    N, M, K = map(int, input().split())

    tests = []   # each entry: (bitmask_of_inserted_keys, opened?)
    for _ in range(M):
        parts = input().split()
        C = int(parts[0])
        keys = list(map(int, parts[1:1 + C]))
        result = parts[-1]          # 'o' or 'x'

        # build bit-mask of the keys used in this test
        mask = 0
        for k in keys:
            mask |= 1 << (k - 1)
        tests.append((mask, result == 'o'))

    answer = 0
    # enumerate every possible real/dummy assignment (2^N possibilities)
    for real in range(1 << N):          # bit i = 1  <=> key (i+1) is real
        ok = True
        for mask, opened in tests:
            cnt = (real & mask).bit_count()     # how many real keys were inserted
            if opened:
                if cnt < K:          # door should have opened but didn't reach K real keys
                    ok = False
                    break
            else:                    # door stayed shut
                if cnt >= K:         # K or more real keys would have opened it
                    ok = False
                    break
        if ok:
            answer += 1

    print(answer)


if __name__ == "__main__":
    main()