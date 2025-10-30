import sys

def main() -> None:
    input = sys.stdin.readline

    N = int(input())
    q = [0] * (N + 1)
    r = [0] * (N + 1)
    for i in range(1, N + 1):
        qi, ri = map(int, input().split())
        q[i] = qi
        r[i] = ri

    Q = int(input())
    answers = []
    for _ in range(Q):
        t, d = map(int, input().split())
        qi = q[t]
        ri = r[t]

        # smallest non-negative increment so that (d + delta) % qi == ri
        delta = (ri - (d % qi)) % qi
        answers.append(str(d + delta))

    print("
".join(answers))

if __name__ == "__main__":
    main()