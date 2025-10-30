def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    ans = [0] * n
    for k in range(n):
        s = A[k]
        left = k
        right = k
        # Repeatedly try to absorb a neighbor on the left and/or right.
        while True:
            changed = False
            if left > 0 and A[left - 1] < s:
                s += A[left - 1]
                left -= 1
                changed = True
            if right < n - 1 and A[right + 1] < s:
                s += A[right + 1]
                right += 1
                changed = True
            if not changed:
                break
        ans[k] = s
    sys.stdout.write(" ".join(map(str, ans)))
    
if __name__ == '__main__':
    main()