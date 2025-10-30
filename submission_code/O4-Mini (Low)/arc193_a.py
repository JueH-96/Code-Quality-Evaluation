import sys
import threading

def main():
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    N = int(input())
    W = list(map(int, input().split()))
    LR = [tuple(map(int, input().split())) for _ in range(N)]
    # We will build two arrays of size 2N+2:
    #   min_left_end[x]  = min W_i over intervals i with R_i < x
    #   min_right_strt[x]= min W_i over intervals i with L_i > x
    M = 2 * N
    INF = 10**20
    min_left_end  = [INF] * (M+3)
    min_right_strt= [INF] * (M+3)
    # First, for each interval i, update at position R_i+1 the value W_i
    for i in range(N):
        L_i, R_i = LR[i]
        w = W[i]
        # it lies entirely to the left of any x > R_i
        # so we record w at index R_i+1 in min_left_end
        ri1 = R_i + 1
        if ri1 <= M+1 and w < min_left_end[ri1]:
            min_left_end[ri1] = w
        # similarly it lies entirely to the right of any x < L_i
        # so we record w at index L_i-1 in min_right_strt
        li1 = L_i - 1
        if li1 >= 0 and w < min_right_strt[li1]:
            min_right_strt[li1] = w

    # Now prefix‐min to fill out min_left_end[x] = min over all y<=x of min_left_end[y]
    for x in range(1, M+2):
        if min_left_end[x-1] < min_left_end[x]:
            min_left_end[x] = min_left_end[x-1]
    # And suffix‐min for min_right_strt
    for x in range(M, -1, -1):
        if min_right_strt[x+1] < min_right_strt[x]:
            min_right_strt[x] = min_right_strt[x+1]

    Q = int(input())
    out = []
    for _ in range(Q):
        s,t = map(int, input().split())
        s -= 1
        t -= 1
        Ls, Rs = LR[s]
        Lt, Rt = LR[t]
        Ws = W[s]
        Wt = W[t]
        # 1) If disjoint, answer = Ws+Wt
        if Rs < Lt or Rt < Ls:
            out.append(str(Ws + Wt))
        else:
            # they overlap; we must pick k entirely left of min(Ls,Lt) or entirely right of max(Rs,Rt)
            left_bound  = min(Ls, Lt)
            right_bound = max(Rs, Rt)
            m1 = min_left_end[left_bound]       # intervals with R < left_bound
            m2 = min_right_strt[right_bound]    # intervals with L > right_bound
            m = min(m1, m2)
            if m >= INF:
                out.append("-1")
            else:
                out.append(str(Ws + Wt + m))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()