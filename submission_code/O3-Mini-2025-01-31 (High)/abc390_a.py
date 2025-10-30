def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    A = list(map(int, data))
    sorted_A = [1, 2, 3, 4, 5]
    
    result = "No"
    for i in range(4):
        B = A[:]  # create a copy of A
        B[i], B[i+1] = B[i+1], B[i]  # swap adjacent elements
        if B == sorted_A:
            result = "Yes"
            break
            
    print(result)

if __name__ == '__main__':
    main()