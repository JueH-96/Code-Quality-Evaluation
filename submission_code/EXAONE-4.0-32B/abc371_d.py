import bisect
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    xs = list(map(int, data[1:1+n]))
    ps = list(map(int, data[1+n:1+2*n]))
    q = int(data[1+2*n])
    
    prefix = [0] * (n+1)
    for i in range(n):
        prefix[i+1] = prefix[i] + ps[i]
    
    queries = []
    index = 1 + 2*n + 1
    for i in range(q):
        L = int(data[index]); R = int(data[index+1]); index += 2
        queries.append((L, R))
    
    out_lines = []
    for (L, R) in queries:
        left_idx = bisect.bisect_left(xs, L)
        right_idx = bisect.bisect_right(xs, R) - 1
        
        if left_idx > right_idx:
            out_lines.append("0")
        else:
            total = prefix[right_idx+1] - prefix[left_idx]
            out_lines.append(str(total))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()