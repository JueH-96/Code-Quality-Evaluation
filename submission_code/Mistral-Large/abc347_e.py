import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1

    queries = list(map(int, data[index:index + Q]))
    index += Q

    A = [0] * N
    S = set()
    current_sum = 0

    for x in queries:
        if x in S:
            S.remove(x)
            current_sum -= 1
        else:
            S.add(x)
            current_sum += 1

        for j in S:
            A[j - 1] += current_sum

    print(" ".join(map(str, A)))

if __name__ == "__main__":
    solve()