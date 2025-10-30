import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = int(next(it))
    A = [int(next(it)) for _ in range(n-1)]
    answer = -1
    for x in range(101):
        S = sorted(A + [x])
        # drop the smallest and largest, sum the rest
        if sum(S[1:-1]) >= X:
            answer = x
            break
    print(answer)

main()