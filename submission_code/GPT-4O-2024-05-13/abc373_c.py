# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N+1]))
    
    max_A = max(A)
    max_B = max(B)
    
    print(max_A + max_B)

if __name__ == "__main__":
    main()