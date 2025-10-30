import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    A = list(map(int, input[idx:idx+N]))
    idx +=N
    W = list(map(int, input[idx:idx+N]))
    
    box_to_weights = defaultdict(list)
    for a, w in zip(A, W):
        box_to_weights[a].append(w)
    
    total =0
    for weights in box_to_weights.values():
        total += sum(weights) - max(weights)
    
    print(total)
    
if __name__ == "__main__":
    main()