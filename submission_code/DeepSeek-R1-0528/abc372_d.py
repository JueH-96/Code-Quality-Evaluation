import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    H = list(map(int, data[1:1+n]))
    
    prev_greater = [-1] * n
    stack = []
    for i in range(n):
        while stack and H[stack[-1]] <= H[i]:
            stack.pop()
        if stack:
            prev_greater[i] = stack[-1]
        else:
            prev_greater[i] = -1
        stack.append(i)
    
    diff = [0] * (n+1)
    
    for j in range(n):
        L = prev_greater[j]
        if L < 0:
            L = 0
        else:
            L = L
        R = j - 1
        if L <= R:
            diff[L] += 1
            if R+1 < n:
                diff[R+1] -= 1
                
    res = []
    current = 0
    for i in range(n):
        current += diff[i]
        res.append(str(current))
        
    print(" ".join(res))

if __name__ == "__main__":
    main()