import sys

def main():
    N = int(input())
    A = [list(map(int, input().split())) for _ in range(N)]
    
    current_element = 1
    for i in range(N):
        if current_element <= i + 1:
            current_element = A[i][current_element - 1]
        else:
            current_element = A[current_element - 1][i]
    
    print(current_element)

if __name__ == "__main__":
    main()