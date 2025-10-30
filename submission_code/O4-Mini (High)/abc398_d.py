import sys
import threading

def main():
    data = sys.stdin.readline().split()
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = sys.stdin.readline().strip()

    # cum_r, cum_c is cumvec[t] after t-th wind
    cum_r = 0
    cum_c = 0

    # cs holds cumvec[s] for all stamp times s (initial s=0 and any regen times)
    cs = set()
    cs.add((0, 0))

    ans = []
    for ch in S:
        # apply wind
        if ch == 'N':
            cum_r -= 1
        elif ch == 'S':
            cum_r += 1
        elif ch == 'W':
            cum_c -= 1
        elif ch == 'E':
            cum_c += 1

        # check if target (R,C) is covered at time t+0.5:
        # need cumvec[s] = cumvec[t] - (R,C) for some stamp s
        key = (cum_r - R, cum_c - C)
        if key in cs:
            ans.append('1')
        else:
            ans.append('0')

        # regen if origin empty: i.e., if cumvec[t] is new among stamps
        pos = (cum_r, cum_c)
        if pos not in cs:
            cs.add(pos)

    sys.stdout.write(''.join(ans))

# call main
if __name__ == "__main__":
    main()