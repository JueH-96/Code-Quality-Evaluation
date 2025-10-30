import sys

def main():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())
    # Read list of H_i
    H = list(map(int, sys.stdin.readline().split()))
    
    remaining = M
    satisfied = 0
    
    for hands in H:
        if remaining >= hands:
            remaining -= hands
            satisfied += 1
        else:
            # Not enough disinfectant for this alien; they use up what's left.
            break
    
    print(satisfied)

if __name__ == "__main__":
    main()