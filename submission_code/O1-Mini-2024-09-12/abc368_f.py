import sys

def main():
    import sys
    import threading

    def solve():
        import sys

        sys.setrecursionlimit(1 << 25)
        N_and_rest = sys.stdin.read().split()
        N = int(N_and_rest[0])
        A = list(map(int, N_and_rest[1:N+1]))

        MAX = 100001
        divisors = [[] for _ in range(MAX)]

        for x in range(1, MAX):
            for m in range(2*x, MAX, x):
                divisors[m].append(x)

        Grundy = [0] * MAX
        for m in range(2, MAX):
            mask = 0
            for x in divisors[m]:
                g = Grundy[x]
                if g < 32:
                    mask |= (1 << g)
            mex = 0
            while mask & (1 << mex):
                mex +=1
            Grundy[m] = mex

        total_xor = 0
        for num in A:
            total_xor ^= Grundy[num]

        print("Anna" if total_xor !=0 else "Bruno")

    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()