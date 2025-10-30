# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    A = []
    
    for i in range(1, N + 1):
        A.append(list(map(int, data[i].split())))
    
    current_element = 1
    
    for j in range(1, N + 1):
        if current_element >= j:
            current_element = A[current_element - 1][j - 1]
        else:
            current_element = A[j - 1][current_element - 1]
    
    print(current_element)

if __name__ == "__main__":
    main()