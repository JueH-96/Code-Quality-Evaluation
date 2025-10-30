# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    result = []
    for b in B:
        eaten = False
        for i in range(N):
            if b >= A[i]:
                result.append(i + 1)
                eaten = True
                break
        if not eaten:
            result.append(-1)
    
    for res in result:
        print(res)

if __name__ == "__main__":
    main()