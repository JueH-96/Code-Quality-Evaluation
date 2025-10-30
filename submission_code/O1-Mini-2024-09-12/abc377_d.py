# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    max_L_for_r = [0] * (M + 2)  # 1-based indexing
    
    idx = 2
    for _ in range(N):
        L = int(data[idx])
        R = int(data[idx+1])
        if L > max_L_for_r[R]:
            max_L_for_r[R] = L
        idx +=2
    
    max_L = 0
    total_forbidden = 0
    for r in range(1, M+1):
        if max_L_for_r[r] > max_L:
            max_L = max_L_for_r[r]
        total_forbidden += max_L
    
    total_pairs = M * (M +1) //2
    answer = total_pairs - total_forbidden
    print(answer)

if __name__ == "__main__":
    main()