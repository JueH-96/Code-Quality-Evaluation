import math
import sys

def main():
    # Read N from input
    N = int(sys.stdin.readline())
    
    # Calculate M
    M = math.ceil(math.log2(N)) if N > 1 else 1
    
    # Assign each friend to a bit position
    # For each friend, list the bottles where the corresponding bit is 1
    friends = []
    for i in range(M):
        bottles = [j+1 for j in range(N) if (j >> i) & 1]
        friends.append(bottles)
    
    # Print M
    print(M, flush=True)
    
    # Print the list of bottles for each friend
    for bottles in friends:
        print(len(bottles), ' '.join(map(str, sorted(bottles))), flush=True)
    
    # Read the response S
    S = sys.stdin.readline().strip()
    
    # Reconstruct the binary number
    value = 0
    for i in range(M):
        value += int(S[i]) * (1 << i)
    
    # Determine the spoiled bottle number
    X = value + 1  # Since bottle numbering starts at 1
    
    # Print the spoiled bottle number
    print(X, flush=True)

if __name__ == "__main__":
    main()