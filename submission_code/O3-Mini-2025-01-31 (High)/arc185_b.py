def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        # Read the array (convert items to int)
        arr = list(map(int, data[index:index+n]))
        index += n
        s = sum(arr)
        # Since allowed operations never decrease A[0] and never increase A[-1],
        # in any achievable final array B we must have B[0] >= A[0] and B[-1] <= A[-1].
        # If B is non-decreasing then its average lies between B[0] and B[-1] so it must lie in [A[0], A[-1]].
        # That is, A[0] <= S/N <= A[-1], i.e. N*A[0] <= S <= N*A[-1].
        if s >= n * arr[0] and s <= n * arr[-1]:
            out_lines.append("Yes")
        else:
            out_lines.append("No")
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()