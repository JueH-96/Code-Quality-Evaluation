# YOUR CODE HERE
import sys

def main():
    import sys
    sys.setrecursionlimit(1<<25)
    N_L_R = sys.stdin.readline().strip().split()
    while len(N_L_R) < 3:
        N_L_R += sys.stdin.readline().strip().split()
    N, L, R = map(int, N_L_R)
    N = int(N)
    L = int(L)
    R = int(R)
    total_sum = 0

    ranges = []

    max_k = N

    while L <= R:
        for k in range(max_k, -1, -1):
            size = 1 << k
            if L % size == 0 and L + size -1 <= R:
                j = L // size
                ranges.append( (k, j) )
                L += size
                break
        else:
            k = 0
            size = 1
            j = L // size
            ranges.append( (k, j) )
            L += size

    for i,j in ranges:
        print(f"? {i} {j}")
        sys.stdout.flush()
        T_line = sys.stdin.readline()
        while T_line.strip() == '':
            T_line = sys.stdin.readline()
        T = int(T_line.strip())
        if T == -1:
            sys.exit()
        else:
            total_sum = (total_sum + T) % 100

    print(f"! {total_sum}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()