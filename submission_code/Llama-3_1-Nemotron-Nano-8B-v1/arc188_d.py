MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(lambda x: int(x) if x != '-1' else -1, input[ptr:ptr+N]))
    ptr += N

    pos_A = {}
    for i in range(N):
        if A[i] in pos_A:
            print(0)
            return
        pos_A[A[i]] = i
    pos_B = {}
    for i in range(N):
        if B[i] != -1:
            if B[i] in pos_B:
                print(0)
                return
            pos_B[B[i]] = i

    for i in range(N):
        if B[i] != -1:
            if not (1 <= B[i] <= 2 * N):
                print(0)
                return
            if A[i] == B[i]:
                print(0)
                return

    reverse_pairs = {}
    used = set()
    for i in range(N):
        if B[i] != -1:
            a = A[i]
            b = B[i]
            if a in used or b in used:
                print(0)
                return
            used.add(a)
            used.add(b)
            reverse_pairs[a] = b

    columns = [[], [], []]
    for i in range(N):
        c = [A[i], B[i]]
        for k in range(3):
            if c[k] == -1:
                c[k] = None
            else:
                columns[k].append(c[k])

    valid = True
    for k in range(3):
        s = set(columns[k])
        for x in s:
            if x is not None and not (1 <= x <= N):
                valid = False
        if not valid:
            break
        if len(s) != len([x for x in columns[k] if x is not None]):
            valid = False
        if not valid:
            break
    if not valid:
        print(0)
        return

    from collections import defaultdict
    cnt = [defaultdict(int) for _ in range(N+1)]
    cnt[0][tuple(columns[0])] = 1

    for k in range(1, 3):
        nk = defaultdict(int)
        for col in columns[k]:
            for prev in cnt[k-1]:
                prev_list = list(prev)
                in_col = True
                for i in range(N):
                    if prev_list[i] is None:
                        continue
                    if col[i] is not None and prev_list[i] != col[i]:
                        in_col = False
                        break
                if not in_col:
                    continue
                new_list = prev_list.copy()
                nk[tuple(new_list)] = (nk[tuple(new_list)] + cnt[k-1][prev]) % MOD
        cnt[k] = nk

    total = cnt[2].get(tuple(columns[2]), 0) % MOD
    print(total)

if __name__ == "__main__":
    main()