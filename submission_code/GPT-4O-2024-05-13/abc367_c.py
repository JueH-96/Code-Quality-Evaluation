# YOUR CODE HERE
import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:2+N]))
    
    ranges = [range(1, r+1) for r in R]
    
    valid_sequences = []
    
    for seq in itertools.product(*ranges):
        if sum(seq) % K == 0:
            valid_sequences.append(seq)
    
    valid_sequences.sort()
    
    for seq in valid_sequences:
        print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()