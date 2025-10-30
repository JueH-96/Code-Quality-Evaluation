# YOUR CODE HERE
import sys

def main():
    N = int(sys.stdin.readline())
    M = 0
    while (1 << M) < N:
        M += 1
    print(M)
    sys.stdout.flush()
    
    # Prepare the bottles for each friend
    friends = []
    for i in range(M):
        bottles = []
        for j in range(1, N+1):
            if (j >> i) & 1:
                bottles.append(j)
        friends.append(bottles)
    
    # Output the bottles for each friend
    for bottles in friends:
        print(len(bottles), end=' ')
        print(' '.join(map(str, bottles)))
        sys.stdout.flush()
    
    # Read the response from the judge
    S = sys.stdin.readline().strip()
    
    # Determine the spoiled bottle
    X = 0
    for i in range(M):
        if S[i] == '1':
            X |= (1 << i)
    
    # Handle the case where X is zero (which should not happen as per constraints)
    if X == 0:
        X = 1
    
    print(X)
    sys.stdout.flush()

if __name__ == "__main__":
    main()