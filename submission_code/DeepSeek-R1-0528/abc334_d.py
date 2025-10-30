import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    q = int(data[1])
    R = list(map(int, data[2:2+n]))
    queries = list(map(int, data[2+n:2+n+q]))
    
    R.sort()
    prefix = [0]
    for i in range(n):
        prefix.append(prefix[-1] + R[i])
    
    out_lines = []
    for x in queries:
        idx = bisect.bisect_right(prefix, x)
        out_lines.append(str(idx - 1))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()