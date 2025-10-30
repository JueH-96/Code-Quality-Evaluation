import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    h = list(map(int, data[1:1+n]))
    
    stack = [-1]
    f = [0] * n
    
    for i in range(n):
        while len(stack) > 1 and h[stack[-1]] < h[i]:
            stack.pop()
        j = stack[-1]
        
        if j == -1:
            f[i] = (i + 1) * (h[i] + 1) - i
        else:
            f[i] = f[j] + (i - j) * h[i]
        
        stack.append(i)
    
    print(" ".join(map(str, f)))

if __name__ == "__main__":
    main()