import sys

def solve():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    c0 = 0
    c1 = 0
    total = 0
    for j in range(1, N+1):
        if S[j-1] == '0':
            c1_j = j-1
            c0_j = 1
            total += c1_j
        else:
            c1_j = c0 + 1
            c0_j = c1
            total += c1_j
        c0, c1 = c0_j, c1_j
    print(total)

if __name__ == "__main__":
    solve()