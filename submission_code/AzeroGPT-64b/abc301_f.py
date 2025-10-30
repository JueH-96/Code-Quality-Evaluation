MOD = 998244353

def matMult(A, B):
    """
    Matrix multiplication for 3x3 matrices.
    """
    return [
        [
            (A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] * B[2][0]) % MOD,
            (A[0][0] * B[0][1] + A[0][1] * B[1][1] + A[0][2] * B[2][1]) % MOD,
            (A[0][0] * B[0][2] + A[0][1] * B[1][2] + A[0][2] * B[2][2]) % MOD,
        ],
        [
            (A[1][0] * B[0][0] + A[1][1] * B[1][0] + A[1][2] * B[2][0]) % MOD,
            (A[1][0] * B[0][1] + A[1][1] * B[1][1] + A[1][2] * B[2][1]) % MOD,
            (A[1][0] * B[0][2] + A[1][1] * B[1][2] + A[1][2] * B[2][2]) % MOD,
        ],
        [
            (A[2][0] * B[0][0] + A[2][1] * B[1][0] + A[2][2] * B[2][0]) % MOD,
            (A[2][0] * B[0][1] + A[2][1] * B[1][1] + A[2][2] * B[2][1]) % MOD,
            (A[2][0] * B[0][2] + A[2][1] * B[1][2] + A[2][2] * B[2][2]) % MOD,
        ],
    ]

def matMult2(A, B):
    """
    Matrix multiplication for 2x2 matrices.
    """
    return [
        [
            (A[0][0] * B[0][0] + A[0][1] * B[1][0]) % MOD,
            (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % MOD,
        ],
        [
            (A[1][0] * B[0][0] + A[1][1] * B[1][0]) % MOD,
            (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % MOD,
        ],
    ]

def matExp(M, pow_val):
    """
    Exponential for 3x3 matrice.
    """
    I = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    while pow_val:
        if pow_val % 2 == 1:
            I = matMult(I, M)
        M = matMult(M, M)
        pow_val = pow_val // 2
    return I

def solve(string):
    st0 = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    st1 = [
        [52, 26, 0],
        [0, 52, 52],
        [0, 0, 52],
    ]
    st2 = [
        [0, 26, 0],
        [0, 0, 52],
    ]
    st3 = [
        [52, 0],
        [0, 52],
    ]
    result = 0
    i = len(string) - 1
    curr2 = matExp(st1, i)
    curr3 = matMult(st2, curr2)
    result += matMult(curr3, st3)[1][1]
    result %= MOD
    while i >= 4:
        c = string[i - 4]
        if c.isupper():
            # if uppercase, the second element could be upper or lower
            # which leads to two matrices that will be added together
            upp = matMult([st0[0]], curr2)
            upper = matMult(upp, st2)
            uppr = matMult(upper, st3)
            low = matMult([st0[1:]], curr2)
            lower = matMult(low, st2)
            loww = matMult(lower, st3)
            result += uppr[0][1] + loww[0][1]
            result %= MOD
            i -= 1
            curr3 = matMult(st1[0:1], curr2)
            curr3 = matMult(curr3, st2)
            curr2 = matMult(st1[1:3], curr2)
        elif c.islower():
            # letter 3 must be lower
            lower = matMult([st0[1:]], curr2)
            lower = matMult(lower, st2)
            low = matMult(lower, st3)
            result += low[0][1]
            result %= MOD
            i -= 1
            curr3 = matMult(st1[1:2], curr2)
            curr3 = matMult(curr3, st2)
            curr2 = matMult(st1[2:], curr2)
        else:
            # *
            result += matMult2(curr3, st3)[1][1]
            result %= MOD
            result += matMult2(curr2[1:2], st2)[0][0]
            result %= MOD
            i -= 1
            curr2 = matMult(st1, curr2)
    return str(result)

if __name__ == "__main__":
    S = input()
    print(solve(S))