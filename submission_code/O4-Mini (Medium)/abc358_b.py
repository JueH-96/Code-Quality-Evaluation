def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = int(data[1])
    T = list(map(int, data[2:]))

    current_end = 0
    out = []
    for t in T:
        finish = max(t, current_end) + A
        out.append(str(finish))
        current_end = finish

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()