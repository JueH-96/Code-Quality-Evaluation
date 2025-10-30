import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n = data[0]
    a = data[1:]

    # next_person[x] = the person standing right behind person x (0 means none)
    next_person = [0] * (n + 1)
    front = -1

    for i, pred in enumerate(a, 1):          # i : current person (1-indexed)
        if pred == -1:                       # person i is at the very front
            front = i
        else:                                # person i is right behind `pred`
            next_person[pred] = i

    order = []
    cur = front
    while cur:
        order.append(cur)
        cur = next_person[cur]

    print(" ".join(map(str, order)))

if __name__ == "__main__":
    main()