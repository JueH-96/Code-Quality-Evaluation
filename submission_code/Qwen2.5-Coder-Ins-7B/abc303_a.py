# YOUR CODE HERE
import sys

def are_similar(x, y):
    return x == y or (x == '1' and y == 'l') or (x == 'l' and y == '1') or (x == '0' and y == 'o') or (x == 'o' and y == '0')

def main():
    N = int(sys.stdin.readline().strip())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    for i in range(N):
        if not are_similar(S[i], T[i]):
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()