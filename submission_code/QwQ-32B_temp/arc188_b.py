import sys

def main():
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    for _ in range(T):
        N = int(input[idx])
        K = int(input[idx+1])
        idx += 2
        if N % 2 == 0:
            if K == N // 2:
                print("No")
                continue
        a = set()
        if N % 2 == 0:
            a.add(0)
            a.add(N // 2)
        else:
            a.add(0)
        b = set()
        if N % 2 == 0:
            b.add(K)
            b.add((K + N // 2) % N)
        else:
            b.add(K)
        union = a.union(b)
        total = len(union)
        not_on = N - total
        print("Yes" if not_on % 2 == 0 else "No")

if __name__ == "__main__":
    main()