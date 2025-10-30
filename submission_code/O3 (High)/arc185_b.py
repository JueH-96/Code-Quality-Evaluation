import sys

# ------------------------------------------------------------
#  Can we make the array non-decreasing only by moving units
#  from a larger index to a smaller one?
#
#  Let
#        S_k = A_1 + … + A_k      (1 ≤ k ≤ N)
#
#  After the moves the new array B has to satisfy
#
#     (i)  B_1 ≤ B_2 ≤ … ≤ B_N          (non-decreasing)
#    (ii)  Σ_{i=1..k} B_i ≥ S_k         (every prefix may only grow)
#   (iii)  Σ_{i=1..N} B_i = S_N         (total sum is unchanged)
#
#  While scanning the array from left to right we maintain
#
#        cum … sum of B’s that are already fixed
#        prev … B_{i-1}                  (value fixed at previous step)
#
#  When we are going to fix B_i, the following must hold
#
#        lower = max(prev, S_i - cum)            (has to meet (i) & (ii))
#
#  Let   rem   = S_N - cum        –  still unassigned total
#        cnt   = N - i + 1        –  positions still to fill
#
#  Because the remaining (cnt-1) positions have to be
#  at least B_i, we must have
#
#        rem - B_i ≥ (cnt-1)·B_i  ⇒  B_i ≤ rem // cnt
#
#        upper = rem // cnt                       (necessary upper bound)
#
#  If lower > upper no integer B_i satisfies all conditions – impossible.
#  Otherwise pick the largest admissible value, B_i = upper.
#  Choosing the largest value guarantees
#
#        upper(i) ≤ upper(i+1)        (easy to verify algebraically)
#
#  hence the non-decreasing property for the whole sequence.
#
#  If we can go through all indices the required array exists,
#  otherwise it does not.
#
#  The algorithm is O(N) per test case, overall
#  O(Σ N) ≤ 2·10⁵.
# ------------------------------------------------------------


def possible(a):
    n = len(a)
    total = sum(a)

    cum = 0               # already fixed part of B
    prev = -10**18        # B_0
    pref = 0              # running prefix of A

    for i, val in enumerate(a, 1):
        pref += val

        remaining = total - cum
        cnt = n - i + 1
        upper = remaining // cnt
        lower = max(prev, pref - cum)

        if lower > upper:           # no admissible value
            return False

        b_i = upper                 # take the largest admissible value
        prev = b_i
        cum += b_i

    return True


def main() -> None:
    it = iter(sys.stdin.read().split())
    t = int(next(it))
    out_lines = []

    for _ in range(t):
        n = int(next(it))
        a = [int(next(it)) for _ in range(n)]
        out_lines.append("Yes" if possible(a) else "No")

    sys.stdout.write("
".join(out_lines))


if __name__ == "__main__":
    main()