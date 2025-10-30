import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    k = int(input[1])
    r_list = list(map(int, input[2:2+n]))
    
    ranges = [range(1, ri + 1) for ri in r_list]
    for seq in itertools.product(*ranges):
        if sum(seq) % k == 0:
            print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()