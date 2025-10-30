# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    Q = int(input[idx+1])
    idx +=2
    R = list(map(int, input[idx:idx+N]))
    idx +=N
    R.sort()
    prefix = [0]*(N+1)
    for i in range(N):
        prefix[i+1] = prefix[i] + R[i]
    for _ in range(Q):
        X = int(input[idx])
        idx +=1
        m = bisect.bisect_right(prefix, X) -1
        print(m)

if __name__ == "__main__":
    main()