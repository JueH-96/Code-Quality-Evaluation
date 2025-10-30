import sys

# ---------------------------------------------
# number of standard Young tableaux of a shape
# ---------------------------------------------
def f_of_shape(rows, mod):
    """
    rows : list with row lengths (partition, non–increasing)
    mod  : prime modulo
    returns  f^λ   ( #SYT of the shape )   modulo mod
    """
    n = sum(rows)                     # number of boxes
    max_len = max(rows)

    # column lengths
    col_len = [0]*(max_len+1)         # 1-based
    for j in range(1, max_len+1):
        col_len[j] = sum(1 for r in rows if r >= j)

    # product of hook lengths
    hook_prod = 1
    for i, row_len in enumerate(rows, 1):          # i : 1-based row index
        for j in range(1, row_len+1):              # j : 1-based column index
            hook = row_len - j                     # cells to the right
            hook += col_len[j] - i                 # cells below
            hook += 1                              # the cell itself
            hook_prod = (hook_prod * hook) % mod

    # n!  modulo mod
    fact = 1
    for k in range(2, n+1):
        fact = (fact * k) % mod

    # f = n! / (product of hooks)  (mod prime)
    inv = pow(hook_prod, mod-2, mod)               # modular inverse (mod is prime)
    return (fact * inv) % mod


def main() -> None:
    A, B, MOD = map(int, sys.stdin.readline().split())

    # shape λ : A  … (B-1 times) …  A,  A-1
    lam = [A]*(B-1) + [A-1]

    # shape μ : λ with the corner cell (B-1, A) removed
    if B == 2:
        mu = [A-1, A-1]
    else:
        mu = [A]*(B-2) + [A-1, A-1]

    f_lambda = f_of_shape(lam, MOD)
    f_mu     = f_of_shape(mu , MOD)

    # answer  = f^λ  *  f^{μ}   (mod MOD)
    ans = (f_lambda * f_mu) % MOD
    print(ans)


if __name__ == "__main__":
    main()