import sys

def main():
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    results = []
    for _ in range(T):
        N = int(input[idx])
        K = int(input[idx+1])
        idx += 2
        if N % 2 == 0:
            half = N // 2
            if K == half:
                if N == 2:
                    results.append("Yes")
                else:
                    results.append("No")
            else:
                results.append("Yes")
        else:
            results.append("Yes")
    print('
'.join(results))

if __name__ == "__main__":
    main()