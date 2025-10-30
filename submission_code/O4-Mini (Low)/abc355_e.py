import sys

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    N, L, R = map(int, data)
    ans = 0
    cur = L
    # While there is still range to cover
    while cur <= R:
        length = R - cur + 1
        # Find largest power of two block starting at cur that fits in [cur,R]
        # and is aligned (cur % block_size == 0).
        best_i = 0
        for i in range(N, -1, -1):
            block = 1 << i
            if block <= length and cur % block == 0:
                best_i = i
                break
        # j = cur // 2^best_i
        j = cur >> best_i
        # Query
        print(f"? {best_i} {j}", flush=True)
        t = int(sys.stdin.readline())
        if t < 0:
            # error
            return
        ans = (ans + t) % 100
        cur += (1 << best_i)
    # Output final answer
    print(f"! {ans}", flush=True)

if __name__ == "__main__":
    main()