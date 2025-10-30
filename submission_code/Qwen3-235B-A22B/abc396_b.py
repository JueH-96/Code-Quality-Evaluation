import sys

def main():
    q = int(sys.stdin.readline())
    stack = [0] * 100
    for _ in range(q):
        query = sys.stdin.readline().split()
        if query[0] == '1':
            x = int(query[1])
            stack.append(x)
        else:
            print(stack.pop())

if __name__ == "__main__":
    main()