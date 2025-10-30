import sys

def find_closest_numbers(N, L, R, A):
    result = []
    for a in A:
        if a < L:
            result.append(L)
        elif a > R:
            result.append(R)
        else:
            result.append(a)
    return result

if __name__ == "__main__":
    input = sys.stdin.read
    N, L, R, *A = map(int, input().split())
    result = find_closest_numbers(N, L, R, A)
    print(" ".join(map(str, result)))