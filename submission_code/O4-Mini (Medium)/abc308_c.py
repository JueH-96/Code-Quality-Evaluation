import sys
import threading

def main():
    import sys
    from functools import cmp_to_key

    input = sys.stdin.readline
    N = int(input())
    people = []
    for i in range(1, N+1):
        A, B = map(int, input().split())
        people.append((A, B, i))

    def cmp(x, y):
        Ax, Bx, ix = x
        Ay, By, iy = y
        # Compare Ax/(Ax+Bx) vs Ay/(Ay+By) via cross multiplication:
        # Ax*(Ay+By) ? Ay*(Ax+Bx)
        lhs = Ax * (Ay + By)
        rhs = Ay * (Ax + Bx)
        if lhs > rhs:
            return -1  # x should come before y
        elif lhs < rhs:
            return 1   # y should come before x
        else:
            # tie: smaller index first
            if ix < iy:
                return -1
            elif ix > iy:
                return 1
            else:
                return 0

    people.sort(key=cmp_to_key(cmp))
    result = [str(person[2]) for person in people]
    print(" ".join(result))

if __name__ == "__main__":
    main()