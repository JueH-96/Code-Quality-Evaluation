# YOUR CODE HERE
import sys

def main():
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    W_X = list(map(int, N_and_rest[1:]))
    bases = [(W_X[i], W_X[i+1]) for i in range(0, 2*N, 2)]
    
    counts = [0]*24
    for W, X in bases:
        for t in range(24):
            local_time = (t + X) %24
            if 9 <= local_time <=17:
                counts[t] += W
    print(max(counts))

if __name__ == "__main__":
    main()