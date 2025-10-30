import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    S = data[0]
    T = data[1]
    # Method 1: match all 3 letters of T as a subsequence in S
    p1 = 0
    t0, t1, t2 = T[0].lower(), T[1].lower(), T[2].lower()
    for c in S:
        if p1 == 0 and c == t0:
            p1 = 1
        elif p1 == 1 and c == t1:
            p1 = 2
        elif p1 == 2 and c == t2:
            p1 = 3
            break

    ok1 = (p1 == 3)

    # Method 2: only if T[2] == 'X', match first two letters
    ok2 = False
    if T[2] == 'X':
        p2 = 0
        for c in S:
            if p2 == 0 and c == t0:
                p2 = 1
            elif p2 == 1 and c == t1:
                p2 = 2
                break
        ok2 = (p2 == 2)

    print("Yes" if (ok1 or ok2) else "No")

if __name__ == "__main__":
    main()