import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = []
    index = 1
    for i in range(N):
        A.append(list(map(int, data[index:index+i+1])))
        index += i + 1
    
    current_element = 1
    for i in range(1, N + 1):
        if current_element <= i:
            current_element = A[i-1][current_element-1]
        else:
            current_element = A[current_element-1][i-1]
    
    print(current_element)

if __name__ == "__main__":
    main()