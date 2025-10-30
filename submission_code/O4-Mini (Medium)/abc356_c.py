import sys

def main():
    input = sys.stdin.readline
    N, M, K = map(int, input().split())
    tests = []
    for _ in range(M):
        data = input().split()
        Ci = int(data[0])
        keys = list(map(int, data[1:1+Ci]))
        R = data[-1]
        # build bitmask of these keys
        mask = 0
        for k in keys:
            mask |= 1 << (k-1)
        should_open = (R == 'o')
        tests.append((mask, should_open))
    
    ans = 0
    # iterate over all subsets of {1..N}
    for s in range(1 << N):
        ok = True
        for mask, should_open in tests:
            # count real keys in this test
            real_cnt = (s & mask).bit_count()
            if should_open:
                if real_cnt < K:
                    ok = False
                    break
            else:
                if real_cnt >= K:
                    ok = False
                    break
        if ok:
            ans += 1
    
    print(ans)

if __name__ == "__main__":
    main()