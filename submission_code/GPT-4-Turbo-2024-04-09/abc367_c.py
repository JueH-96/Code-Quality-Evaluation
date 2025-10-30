import sys
import itertools

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:2+N]))
    
    # Generate all possible sequences
    ranges = [range(1, r+1) for r in R]
    all_sequences = itertools.product(*ranges)
    
    # Filter sequences whose sum is a multiple of K
    valid_sequences = [seq for seq in all_sequences if sum(seq) % K == 0]
    
    # Output all valid sequences in lexicographical order
    for seq in valid_sequences:
        print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()