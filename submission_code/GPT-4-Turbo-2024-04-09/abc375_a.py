# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    count = 0
    for i in range(1, N - 1):
        if S[i - 1] == '#' and S[i] == '.' and S[i + 1] == '#':
            count += 1
    
    print(count)

if __name__ == "__main__":
    main()