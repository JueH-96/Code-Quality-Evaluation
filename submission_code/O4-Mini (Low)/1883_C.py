import sys
import threading
def main():
    import sys
    input = sys.stdin.readline

    t = int(input())
    for _ in range(t):
        n,k = map(int,input().split())
        a = list(map(int,input().split()))
        # handle k prime cases 2,3,5: need one factor p
        if k in (2,3,5):
            p = k
            # check if already divisible
            ok = False
            for x in a:
                if x % p == 0:
                    ok = True
                    break
            if ok:
                print(0)
                continue
            # else find minimal t s.t. (x+t)%p==0
            best = p  # at most p-1
            for x in a:
                r = x % p
                t_need = (p - r) % p
                if t_need > 0 and t_need < best:
                    best = t_need
            print(best)
        else:
            # k==4: need total exponent of 2 at least 2
            # compute current sum of exp2
            def exp2(x):
                c=0
                while x&1==0:
                    c+=1
                    x//=2
                return c
            cur = 0
            for x in a:
                cur += exp2(x)
            need = max(0, 2 - cur)
            if need == 0:
                print(0)
                continue
            # prepare gains
            cnt_g1 = 0  # number of elements where gain1>=1
            # track if any single item gives gain>=2
            best_single = None
            # also track if any g2>=1 for completeness
            any_g2_ge1 = False
            for x in a:
                e0 = exp2(x)
                # gain after 1 increment
                e1 = exp2(x+1)
                g1 = e1 - e0
                # gain after 2 increments
                e2 = exp2(x+2)
                g2 = e2 - e0
                if g1 >= 1:
                    cnt_g1 += 1
                # single item of cost1 gives gain>=2?
                if g1 >= 2:
                    # cost =1
                    if best_single is None or 1 < best_single:
                        best_single = 1
                # cost2 item gives gain>=2?
                if g2 >= 2:
                    if best_single is None or 2 < best_single:
                        best_single = 2
                # for completeness check g2>=1
                if g2 >= 1:
                    any_g2_ge1 = True
            # now compute answer for need=1 or 2
            if need == 1:
                # best is cost1 if any g1>=1, else cost2 if any_g2_ge1
                if cnt_g1 > 0:
                    print(1)
                else:
                    # must use a 2-step to get at least 1 gain
                    print(2)
            else:
                # need==2
                ans = 10**18
                # use single item if gives >=2
                if best_single is not None:
                    ans = min(ans, best_single)
                # or use two cost1 items if at least two with g1>=1
                if cnt_g1 >= 2:
                    ans = min(ans, 2)
                # if still inf, maybe use one g1 and one g2>=1: cost=1+2
                if cnt_g1 >= 1 and any_g2_ge1:
                    ans = min(ans, 3)
                # worst case maybe two g2 items: cost4 but probably not needed
                if ans == 10**18:
                    # as fallback try cost=4
                    ans = 4
                print(ans)

if __name__ == "__main__":
    main()