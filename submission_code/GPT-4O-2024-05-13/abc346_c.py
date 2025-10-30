# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = set(map(int, data[2:2+N]))
    
    total_sum = K * (K + 1) // 2
    present_sum = sum(a for a in A if a <= K)
    
    result = total_sum - present_sum
    print(result)

if __name__ == "__main__":
    main()