# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = int(data[1])
    B = int(data[2])
    D = list(map(int, data[3:]))
    
    week_length = A + B
    
    for d in D:
        if (d % week_length) > A or (d % week_length) == 0:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()