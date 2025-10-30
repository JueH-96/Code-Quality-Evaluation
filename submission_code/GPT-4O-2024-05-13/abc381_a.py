# YOUR CODE HERE
import sys
input = sys.stdin.read

def is_1122_string(N, S):
    if N % 2 == 0:
        return "No"
    
    mid = (N + 1) // 2
    if S[mid - 1] != '/':
        return "No"
    
    for i in range(mid - 1):
        if S[i] != '1':
            return "No"
    
    for i in range(mid, N):
        if S[i] != '2':
            return "No"
    
    return "Yes"

def main():
    data = input().strip().split()
    N = int(data[0])
    S = data[1]
    print(is_1122_string(N, S))

if __name__ == "__main__":
    main()