def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    target = "abc"
    out = []
    for _ in range(t):
        s = data[idx]
        idx += 1
        # If already "abc", it's fine.
        if s == target:
            out.append("YES")
            continue
        # Try swapping any two positions once.
        possible = False
        lst = list(s)
        for i in range(3):
            for j in range(i+1, 3):
                lst[i], lst[j] = lst[j], lst[i]
                if "".join(lst) == target:
                    possible = True
                # swap back
                lst[i], lst[j] = lst[j], lst[i]
                if possible:
                    break
            if possible:
                break
        out.append("YES" if possible else "NO")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()