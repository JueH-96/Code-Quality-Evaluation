def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    results = []
    # We simulate at most one swap operation; if already "abc" then it's okay.
    for i in range(1, t + 1):
        s = data[i].strip()
        if s == "abc":
            results.append("YES")
            continue
        possible = False
        # Try all possible two-card swap operations (there are only 3 possible swaps)
        s_list = list(s)
        for i in range(3):
            for j in range(i + 1, 3):
                arr = list(s_list)
                arr[i], arr[j] = arr[j], arr[i]
                if "".join(arr) == "abc":
                    possible = True
                    break
            if possible:
                break
        results.append("YES" if possible else "NO")
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()