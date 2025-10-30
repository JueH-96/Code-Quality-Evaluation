import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    stack = []
    for a in A:
        stack.append(a)
        while len(stack) >=2 and stack[-1] == stack[-2]:
            combined = stack.pop() +1
            stack.pop()
            stack.append(combined)
    print(len(stack))

if __name__ == "__main__":
    main()