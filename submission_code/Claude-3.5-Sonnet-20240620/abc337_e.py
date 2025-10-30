# YOUR CODE HERE
import sys
import math

def read_int():
    return int(sys.stdin.readline().strip())

def print_line(*args):
    print(*args, flush=True)

def main():
    N = read_int()
    M = math.ceil(math.log2(N))
    print_line(M)

    for i in range(M):
        bottles = [j for j in range(1, N+1) if (j-1) & (1 << i)]
        print_line(len(bottles), *bottles)

    S = sys.stdin.readline().strip()
    spoiled = 1
    for i, result in enumerate(S):
        if result == '1':
            spoiled += 1 << i

    print_line(spoiled)

if __name__ == "__main__":
    main()