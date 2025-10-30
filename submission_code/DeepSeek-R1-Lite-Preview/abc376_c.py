def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N]))
    
    A.sort()
    B.sort()
    
    def is_possible(x):
        boxes = B + [x]
        boxes.sort()
        for i in range(N):
            if A[i] > boxes[i]:
                return False
        return True
    
    left = 1
    right = max(A)
    answer = -1
    while left <= right:
        mid = (left + right) // 2
        if is_possible(mid):
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    
    print(answer)

if __name__ == '__main__':
    main()