import sys
import bisect

def main():
    import sys
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    X = list(map(int, input[idx:idx+N])); idx +=N
    P = list(map(int, input[idx:idx+N])); idx +=N
    prefix = [0]*(N+1)
    for i in range(N):
        prefix[i+1] = prefix[i] + P[i]
    Q = int(input[idx]); idx +=1
    output = []
    for _ in range(Q):
        L = int(input[idx]); idx +=1
        R = int(input[idx]); idx +=1
        left = bisect.bisect_left(X, L)
        right = bisect.bisect_right(X, R)
        total = prefix[right] - prefix[left]
        output.append(str(total))
    print('
'.join(output))

if __name__ == "__main__":
    main()