def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    A = list(map(int, input().split()))
    
    is_strictly_increasing = True
    for i in range(N - 1):
        if A[i] >= A[i+1]:
            is_strictly_increasing = False
            break
            
    if is_strictly_increasing:
        print("Yes")
    else:
        print("No")
        
if __name__ == "__main__":
    main()