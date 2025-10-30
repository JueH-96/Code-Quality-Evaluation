def main():
    N = int(input().strip())
    A = [list(map(int, input().split())) for _ in range(N)]
    
    current_element = 1
    for i in range(N):
        current_element = A[max(current_element, i+1) - 1][min(current_element, i+1) - 1]
    
    print(current_element)

if __name__ == "__main__":
    main()