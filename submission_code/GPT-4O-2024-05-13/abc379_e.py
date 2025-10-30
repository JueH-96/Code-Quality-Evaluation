# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    S = data[1]
    
    total_sum = 0
    
    for i in range(N):
        num = 0
        for j in range(i, N):
            num = num * 10 + int(S[j])
            total_sum += num
    
    print(total_sum)

if __name__ == "__main__":
    main()