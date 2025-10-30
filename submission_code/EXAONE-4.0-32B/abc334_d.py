import sys
import bisect

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    q = int(data[1])
    R = list(map(int, data[2:2+n]))
    queries = list(map(int, data[2+n:2+n+q]))
    
    R.sort()
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + R[i - 1]
    
    results = []
    for x in queries:
        k = bisect.bisect_right(prefix, x) - 1
        results.append(str(k))
    
    print("
".join(results))

if __name__ == "__main__":
    main()