import sys

def main():
    input = sys.stdin.readline
    N, M, K = map(int, input().split())
    tests = []
    for _ in range(M):
        data = input().split()
        C = int(data[0])
        keys = list(map(int, data[1:1+C]))
        R = data[-1]
        # build a bitmask for the tested keys
        mask = 0
        for k in keys:
            mask |= 1 << (k-1)
        tests.append((mask, R))
    
    ans = 0
    # iterate over all 2^N assignments of real/dummy
    for assignment in range(1 << N):
        ok = True
        for mask, R in tests:
            # count how many real keys used in this test
            real_count = (assignment & mask).bit_count()
            opened = (real_count >= K)
            if opened and R == 'x':  # contradiction
                ok = False
                break
            if not opened and R == 'o':  # contradiction
                ok = False
                break
        if ok:
            ans += 1
    
    print(ans)

if __name__ == "__main__":
    main()