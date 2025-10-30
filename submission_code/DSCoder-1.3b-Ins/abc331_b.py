N, S, M, L = map(int, input().split())

def min_eggs(N, S, M, L):
    if N % 6 == 0:
        return N // 6 * S
    elif N % 8 == 0:
        return N // 8 * M
    elif N % 12 == 0:
        return N // 12 * L
    else:
        return -1

print(min_eggs(N, S, M, L))