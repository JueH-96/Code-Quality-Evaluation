# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    S = data[1]
    Q = int(data[2])
    
    operations = data[3:]
    
    for i in range(Q):
        c_i = operations[2 * i]
        d_i = operations[2 * i + 1]
        S = S.replace(c_i, d_i)
    
    print(S)

if __name__ == "__main__":
    main()