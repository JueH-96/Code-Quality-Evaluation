def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    pos = 1
    out = []
    for _ in range(t):
        n = int(data[pos])
        pos += 1
        arr = list(map(int, data[pos:pos+n]))
        pos += n
        # dp will store the max sum of an alternating subarray ending at the current index.
        dp = arr[0]
        best = dp
        for i in range(1, n):
            # We try extending the previous subarray if the parity condition holds.
            # Determine parities using modulo: in Python, negative numbers modulo 2 return a non-negative remainder.
            if (arr[i-1] % 2) != (arr[i] % 2):
                dp = max(arr[i], dp + arr[i])
            else:
                dp = arr[i]
            best = max(best, dp)
        out.append(str(best))
    sys.stdout.write("
".join(out))

if __name__ == '__main__':
    main()