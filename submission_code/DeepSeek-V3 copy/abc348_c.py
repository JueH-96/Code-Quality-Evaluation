# YOUR CODE HERE
import sys

def main():
    N = int(sys.stdin.readline())
    color_to_min_a = {}
    
    for _ in range(N):
        A, C = map(int, sys.stdin.readline().split())
        if C in color_to_min_a:
            if A < color_to_min_a[C]:
                color_to_min_a[C] = A
        else:
            color_to_min_a[C] = A
    
    max_min_a = max(color_to_min_a.values())
    print(max_min_a)

if __name__ == "__main__":
    main()