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
        if (2 * K) % N != 0:
            results.append("Yes")
        else:
            results.append("No")
    print('
'.join(results))

if __name__ == "__main__":
    main()