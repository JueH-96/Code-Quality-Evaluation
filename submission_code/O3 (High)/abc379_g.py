import sys
import numpy as np

MOD = 998_244_353


def read_input() -> tuple[int, int, list[list[str]]]:
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    return H, W, grid


def orient_grid(H: int, W: int, g: list[list[str]]):
    # make the shorter side the number of rows  (R)
    if H <= W:
        return H, W, g                     # already fine
    # transpose
    transposed = [[g[r][c] for r in range(H)] for c in range(W)]
    return W, H, transposed


def build_powers(R: int):
    p = [1]
    for _ in range(R):
        p.append(p[-1] * 3)
    return p


def row_transform(dp: np.ndarray, size_before: int):
    """one Kronecker factor:  (x0,x1,x2)->(x1+x2, x0+x2, x0+x1)"""
    view = dp.reshape(size_before, 3, -1)
    total = (view.sum(axis=1) % MOD)           # shape (size_before, -1)
    view[:, 0, :] = (total - view[:, 0, :]) % MOD
    view[:, 1, :] = (total - view[:, 1, :]) % MOD
    view[:, 2, :] = (total - view[:, 2, :]) % MOD
    # reshape back by the caller


def multiply_M(dp: np.ndarray, R: int, pow3: list[int]):
    """dp <- (M⊗…⊗M) · dp   (in place)"""
    for i in range(R):
        row_transform(dp, pow3[i])
    return dp


def build_digit_arrays(R: int, pow3: list[int], total: int):
    base = np.arange(total, dtype=np.int64)
    digit = []
    for i in range(R):
        digit.append(((base // pow3[i]) % 3).astype(np.uint8))
    return digit


def build_vertical_mask(R: int, digit: list[np.ndarray], total: int):
    if R == 1:
        return np.ones(total, dtype=bool)
    mask = digit[0] != digit[1]
    for i in range(1, R - 1):
        mask &= (digit[i] != digit[i + 1])
    return mask


def column_mask(constraints, digit, vertical_mask, total):
    mask = vertical_mask.copy()
    for r, need in constraints:
        mask &= (digit[r] == need)
    return mask


def main() -> None:
    sys.setrecursionlimit(1 << 25)

    H, W, G = read_input()
    R, C, G = orient_grid(H, W, G)          # R rows (≤14), C columns

    pow3 = build_powers(R)
    TOTAL = pow3[R]

    # build digit arrays and vertical-adjacency mask
    digit = build_digit_arrays(R, pow3, TOTAL)
    vert_mask = build_vertical_mask(R, digit, TOTAL)

    # constraints per column
    col_cons = [[] for _ in range(C)]
    for r in range(R):
        for c in range(C):
            ch = G[r][c]
            if ch != '?':
                col_cons[c].append((r, int(ch) - 1))   # convert to 0-based colour

    # dp vector (size TOTAL)
    dp = np.zeros(TOTAL, dtype=np.int64)

    # first column
    first_mask = column_mask(col_cons[0], digit, vert_mask, TOTAL)
    dp[first_mask] = 1

    # remaining columns
    for col in range(1, C):
        multiply_M(dp, R, pow3)                          # dp = T · dp
        mask = column_mask(col_cons[col], digit, vert_mask, TOTAL)
        dp[~mask] = 0                                    # enforce fixed digits

    ans = int(dp.sum() % MOD)
    print(ans)


if __name__ == "__main__":
    main()