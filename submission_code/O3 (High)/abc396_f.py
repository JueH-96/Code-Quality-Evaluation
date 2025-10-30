import sys


# ---------- Fenwick Tree ----------
class Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, idx: int, delta: int = 1):
        while idx <= self.n:
            self.bit[idx] += delta
            idx += idx & -idx

    def pref_sum(self, idx: int) -> int:
        res = 0
        while idx:
            res += self.bit[idx]
            idx -= idx & -idx
        return res


# ---------- Main ----------
def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    # group indices (1-based) by k where they wrap
    wrap_lists = [[] for _ in range(M)]
    for idx, a in enumerate(A, 1):                       # idx : 1-based position
        k_wrap = (M - 1 - a)            # 0 ≤ k_wrap < M  (since 0 ≤ a < M)
        wrap_lists[k_wrap].append(idx)

    # inversion number for k = 0
    bit = Fenwick(M)                     # values are 0 … M-1 => index them by +1
    inv0 = 0
    processed = 0
    for a in A:
        v = a + 1                       # 1-based for Fenwick
        inv0 += processed - bit.pref_sum(v)
        bit.add(v, 1)
        processed += 1

    answers = [0] * M
    answers[0] = inv0
    current = inv0

    for k in range(M - 1):              # compute answers[1] … answers[M-1]
        lst = wrap_lists[k]
        t = len(lst)
        if t:                           # only if something wraps
            s_pos = sum(lst)            # Σ positions in this list
            delta = 2 * (s_pos - t * (t + 1) // 2) - t * (N - t)
            current += delta
        answers[k + 1] = current

    sys.stdout.write('
'.join(map(str, answers)))


if __name__ == "__main__":
    main()