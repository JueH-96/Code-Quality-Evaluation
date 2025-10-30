import sys
import threading

def main():
    import sys
    data = sys.stdin
    T_line = data.readline().strip()
    if not T_line:
        return
    T = int(T_line)
    out_lines = []
    # We will reuse arrays of max needed size to avoid reallocation.
    # But since sum of N <=2e5, we can allocate per test as well.
    for _ in range(T):
        line = data.readline().strip()
        while line == "":
            line = data.readline().strip()
        N = int(line)
        P_list = data.readline().split()
        # build 1-based P
        P = [0] * (N + 2)
        for i, v in enumerate(P_list, start=1):
            P[i] = int(v)
        # check if already sorted (identity)
        sorted_flag = True
        for i in range(1, N+1):
            if P[i] != i:
                sorted_flag = False
                break
        if sorted_flag:
            out_lines.append("0")
            continue
        # build prefix max and suffix min
        pref_max = [0] * (N + 2)
        for i in range(1, N+1):
            # maximum of P[1..i]
            v = P[i]
            m = pref_max[i-1]
            if v > m:
                m = v
            pref_max[i] = m
        INF = N + 1
        suf_min = [INF] * (N + 3)
        suf_min[N+1] = INF
        for i in range(N, 0, -1):
            v = P[i]
            mn = suf_min[i+1]
            if v < mn:
                mn = v
            suf_min[i] = mn
        # check one-op possible
        one_op = False
        # for k from 1 to N
        for k in range(1, N+1):
            if P[k] != k:
                continue
            # prefix max on [1..k-1] < k
            if pref_max[k-1] >= k:
                continue
            # suffix min on [k+1..N] > k
            if suf_min[k+1] <= k:
                continue
            one_op = True
            break
        if one_op:
            out_lines.append("1")
            continue
        # if first element N and last element 1 => need 3 ops
        if P[1] == N and P[N] == 1:
            out_lines.append("3")
        else:
            out_lines.append("2")
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()