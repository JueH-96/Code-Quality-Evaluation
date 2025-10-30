import sys

def solve(N):
    result = []
    for i in range(N + 1):
        found = False
        for j in range(1, 10):
            if N % j == 0 and i % (N // j) == 0:
                result.append(str(j))
                found = True
                break
        if not found:
            result.append('-')
    return ''.join(result)

if __name__ == "__main__":
    N = int(input().strip())
    print(solve(N))