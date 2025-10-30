import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    stack = []
    for a in A:
        if stack and stack[-1] == a:
            stack.pop()
        else:
            stack.append(a)
    
    print(len(stack))

if __name__ == "__main__":
    main()