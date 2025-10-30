import sys
import threading

def main():
    import sys
    data = sys.stdin
    n_line = data.readline().strip()
    if not n_line:
        return
    n = int(n_line)
    # Map each color to the minimum deliciousness seen so far
    color_min = {}
    for _ in range(n):
        a_c = data.readline().split()
        if not a_c:
            continue
        a = int(a_c[0])
        c = int(a_c[1])
        if c in color_min:
            # update the minimum for this color
            if a < color_min[c]:
                color_min[c] = a
        else:
            color_min[c] = a

    # Now find the maximum among those per-color minima
    answer = 0
    for v in color_min.values():
        if v > answer:
            answer = v

    print(answer)

if __name__ == "__main__":
    main()