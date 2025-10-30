import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    Q = int(input[idx+1])
    idx +=2
    S = input[idx]
    idx +=1
    prefix = [0]*(N+1)
    for i in range(1, N):
        prefix[i] = prefix[i-1] + (1 if S[i-1] == S[i] else 0)
    results = []
    for _ in range(Q):
        l = int(input[idx])
        r = int(input[idx+1])
        idx +=2
        if l > r-1:
            results.append("0")
        else:
            count = prefix[r-1] - prefix[l-1]
            results.append(str(count))
    print('
'.join(results))

if __name__ == "__main__":
    main()