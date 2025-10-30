import sys

def solve():
    input = sys.stdin.read
    data = input().split()

    S = data[0]
    Q = int(data[1])
    queries = list(map(int, data[2:]))

    def find_character(k):
        length = len(S)
        while k > length:
            if k <= 2 * length:
                k = k - length
                if k <= length:
                    return S[k-1].swapcase()
                else:
                    return S[k-length-1]
            k = (k + 1) // 2
        return S[k-1]

    results = [find_character(k) for k in queries]
    print(" ".join(results))

solve()