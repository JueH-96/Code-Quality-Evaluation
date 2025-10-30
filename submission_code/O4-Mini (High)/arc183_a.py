import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    # availList holds the current set of values
    avail = list(range(1, N+1))
    # result list
    res = []
    def generate(avail):
        M = len(avail)
        # If there is only one value, output it K times and stop
        if M == 1:
            v = avail[0]
            res.extend([v] * K)
            return
        # If M is odd, pick the middle value K times, remove it, recurse once
        if M & 1:
            j0 = M // 2
            v0 = avail[j0]
            res.extend([v0] * K)
            # build new list without the chosen element
            # M>1 here guaranteed
            new_avail = avail[:j0] + avail[j0+1:]
            # now new_avail has even length => will hit the even branch
            generate(new_avail)
        else:
            # M is even: pick the ceil(M/2)-th (1-based) element once, then
            # exhaust in descending order
            # The local "middle" index (1-based M/2) is zero-based j0 = M//2 - 1
            j0 = M//2 - 1
            v0 = avail[j0]
            # first pick
            res.append(v0)
            # now counts: v0 has K-1 left, others have K
            # output blocks in descending order
            for y in reversed(avail):
                if y == v0:
                    cnt = K - 1
                else:
                    cnt = K
                if cnt > 0:
                    # extend the block for y
                    res.extend([y] * cnt)
            # stop recursion
            return

    if N > 0:
        generate(avail)
    # print result
    # res should have length N*K
    out = ' '.join(map(str, res))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()