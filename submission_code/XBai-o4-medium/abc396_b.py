import sys

def main():
    Q = int(sys.stdin.readline())
    stack = [0] * 100  # Initial stack with 100 zeros
    output = []
    
    for _ in range(Q):
        query = sys.stdin.readline().split()
        if query[0] == '1':
            x = int(query[1])
            stack.append(x)
        else:
            output_val = stack.pop()
            output.append(str(output_val))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()