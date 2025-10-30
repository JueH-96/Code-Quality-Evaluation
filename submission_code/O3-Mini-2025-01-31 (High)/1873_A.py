def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    t = int(data[0])
    results = []
    index = 1
    for _ in range(t):
        s = data[index]
        index += 1

        # If the configuration is already "abc", no swap is needed.
        if s == "abc":
            results.append("YES")
            continue

        found = False
        # Try swapping every possible pair of indices (i, j) once.
        for i in range(3):
            for j in range(i + 1, 3):
                temp = list(s)
                temp[i], temp[j] = temp[j], temp[i]
                if "".join(temp) == "abc":
                    found = True
                    break
            if found:
                break

        results.append("YES" if found else "NO")

    sys.stdout.write("
".join(results))


if __name__ == '__main__':
    main()