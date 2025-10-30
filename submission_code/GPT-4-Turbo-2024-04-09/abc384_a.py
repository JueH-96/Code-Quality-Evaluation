# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    c1 = data[1]
    c2 = data[2]
    S = data[3]
    
    result = ''.join([c if c == c1 else c2 for c in S])
    print(result)

if __name__ == "__main__":
    main()