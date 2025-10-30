# YOUR CODE HERE
import sys
def main():
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A_list = N_and_rest[1:]
    counts = [0] * (N + 1)
    f = [0] * (N + 1)
    j = 1
    for A_j in A_list:
        x = int(A_j)
        counts[x] +=1
        if counts[x] == 2:
            f[x] = j
        j +=1
    fi_list = []
    for i in range(1, N +1):
        fi_list.append( (f[i], i))
    fi_list.sort()
    result = [str(i[1]) for i in fi_list]
    print(' '.join(result))

if __name__ == "__main__":
    main()