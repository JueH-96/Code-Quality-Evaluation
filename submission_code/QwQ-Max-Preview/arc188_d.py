MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    idx += N

    used = set()
    for a in A:
        if a in used:
            print(0)
            return
        used.add(a)
    B_non_neg = []
    for b in B:
        if b != -1:
            if b in used or (b in B_non_neg):
                print(0)
                return
            used.add(b)
            B_non_neg.append(b)
        else:
            B_non_neg.append(-1)
    B = B_non_neg

    missing = []
    for num in range(1, 2*N + 1):
        if num not in used:
            missing.append(num)
    M = len(missing)
    K = B.count(-1)
    if M != K:
        print(0)
        return

    from itertools import permutations

    def generate_B():
        indices = [i for i in range(N) if B[i] == -1]
        for perm in permutations(missing):
            new_B = B.copy()
            for idx, val in zip(indices, perm):
                new_B[idx] = val
            yield new_B

    total = 0
    for b in generate_B():
        a = A
        valid = True
        ab = a + b
        if len(set(ab)) != 2*N:
            continue
        for i in range(N):
            ai = a[i]
            bi = b[i]
            if ai == bi:
                valid = False
                break
        if not valid:
            continue

        cnt = 1
        for i in range(N):
            ai = a[i]
            bi = b[i]
            if (ai < bi) == (bi > ai):
                pass
            else:
                valid = False
                break
        if not valid:
            continue

        total = (total + cnt) % MOD

    print(total % MOD)

if __name__ == '__main__':
    main()