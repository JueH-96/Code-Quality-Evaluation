import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    t = int(next(it))
    out = []
    for _ in range(t):
        n = int(next(it)); k = int(next(it))
        a = [int(next(it)) for _ in range(n)]
        # Handle based on k
        if k == 2:
            # Need one even in product
            ok = False
            for x in a:
                if x % 2 == 0:
                    ok = True
                    break
            if ok:
                out.append("0")
            else:
                # any odd needs 1 op
                out.append("1")
        elif k == 3:
            # Need one divisible by 3
            ok = False
            best = 10**18
            for x in a:
                r = x % 3
                if r == 0:
                    ok = True
                else:
                    best = min(best, 3 - r)
            if ok:
                out.append("0")
            else:
                out.append(str(best))
        elif k == 5:
            # Need one divisible by 5
            ok = False
            best = 10**18
            for x in a:
                r = x % 5
                if r == 0:
                    ok = True
                else:
                    best = min(best, 5 - r)
            if ok:
                out.append("0")
            else:
                out.append(str(best))
        else:  # k == 4
            # Need total v2 >= 2
            # compute current sum
            total_v2 = 0
            for x in a:
                # count v2(x)
                c = (x & -x).bit_length() - 1 if x != 0 else 0
                # above trick isn't quite right; simpler:
                # we want the exponent of 2 dividing x
                # so do:
                v = 0
                y = x
                while y & 1 == 0:
                    v += 1
                    y >>= 1
                total_v2 += v
            if total_v2 >= 2:
                out.append("0")
            else:
                INF = 10**18
                # cost to make v2>=1
                cost1 = []
                c2 = INF
                for x in a:
                    # cost to become even
                    c1 = x % 2  # 0 if even, 1 if odd
                    cost1.append(c1)
                    # cost to become divisible by 4
                    r4 = x % 4
                    c_2 = (4 - r4) % 4
                    c2 = min(c2, c_2)
                # best two distinct for v2>=1
                cost1.sort()
                best_pair = INF
                if len(cost1) >= 2:
                    best_pair = cost1[0] + cost1[1]
                ans = min(c2, best_pair)
                out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()