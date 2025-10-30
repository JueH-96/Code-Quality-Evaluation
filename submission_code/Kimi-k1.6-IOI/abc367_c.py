import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    k = int(input[1])
    r_list = list(map(int, input[2:2+n]))
    
    ranges = [range(1, r+1) for r in r_list]
    sequences = itertools.product(*ranges)
    
    valid = []
    for seq in sequences:
        if sum(seq) % k == 0:
            valid.append(seq)
    
    for seq in valid:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()