import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))

    # Read tests: build a bitmask for each test and store the result character
    masks = []
    results = []
    for _ in range(M):
        Ci = int(next(it))
        mask = 0
        for __ in range(Ci):
            key = int(next(it)) - 1
            mask |= (1 << key)
        res = next(it)
        masks.append(mask)
        results.append(res)

    ans = 0
    # Enumerate all possible assignments of real/dummy keys (2^N)
    for assignment in range(1 << N):
        ok = True
        # Check every test
        for mask, res in zip(masks, results):
            # count how many real keys in this test
            real_count = (assignment & mask).bit_count()
            if res == 'o':
                # door must open: at least K real keys
                if real_count < K:
                    ok = False
                    break
            else:  # res == 'x'
                # door must not open: fewer than K real keys
                if real_count >= K:
                    ok = False
                    break
        if ok:
            ans += 1

    print(ans)


if __name__ == "__main__":
    main()