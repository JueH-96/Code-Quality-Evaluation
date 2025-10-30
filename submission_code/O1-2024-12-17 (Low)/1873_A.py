def main():
    import sys
    t = int(sys.stdin.readline().strip())
    for _ in range(t):
        s = sys.stdin.readline().strip()
        if s == "abc":
            print("YES")
            continue
        # Try all pairs of indices to swap
        found = False
        arr = list(s)
        for i in range(3):
            for j in range(i+1, 3):
                arr[i], arr[j] = arr[j], arr[i]
                if "".join(arr) == "abc":
                    found = True
                # Swap back
                arr[i], arr[j] = arr[j], arr[i]
        print("YES" if found else "NO")

# Don't forget to call main
if __name__ == "__main__":
    main()