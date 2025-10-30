import sys

def main():
    import sys
    import sys
    def input():
        return sys.stdin.read()

    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    last_greater = [0]*(N+1)  # 1-based indexing
    stack = []
    
    for j in range(1, N+1):
        while stack and H[stack[-1]-1] < H[j-1]:
            stack.pop()
        last_greater[j] = stack[-1] if stack else 0
        stack.append(j)
    
    diff = [0]*(N+2)
    
    for j in range(1, N+1):
        left = last_greater[j] if last_greater[j] >0 else 1
        right = j -1
        if left <= right:
            diff[left] +=1
            diff[right +1] -=1
    
    res = []
    current = 0
    for i in range(1, N+1):
        current += diff[i]
        res.append(str(current))
    
    print(' '.join(res))

if __name__ == "__main__":
    main()