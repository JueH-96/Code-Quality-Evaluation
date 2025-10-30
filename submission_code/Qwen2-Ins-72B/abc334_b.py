A, M, L, R = map(int, input().split())

def ceildiv(a, b):
    return -(-a // b)

left = ceildiv(A - L, M)
right = ceildiv(R - A, M)

print(max(0, right - left + 1))