def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    target = "abc"
    idx = 1
    for _ in range(t):
        s = data[idx]
        idx += 1
        # Count positions where s differs from "abc"
        diff = sum(1 for i in range(3) if s[i] != target[i])
        # We can fix with at most one swap iff diff is 0 or 2
        if diff <= 2:
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()