import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    N, T = map(int, input().split())
    # Current scores of players 1..N (0-indexed)
    scores = [0] * N
    # Count of how many players have a given score
    count = {0: N}
    distinct = 1

    out = []
    for _ in range(T):
        a, b = map(int, input().split())
        p = a - 1
        old = scores[p]
        new = old + b

        # Decrement old score count
        cnt_old = count[old]
        if cnt_old == 1:
            # removing the last player with this score
            del count[old]
            distinct -= 1
        else:
            count[old] = cnt_old - 1

        # Increment new score count
        if new in count:
            count[new] += 1
        else:
            count[new] = 1
            distinct += 1

        scores[p] = new
        out.append(str(distinct))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()