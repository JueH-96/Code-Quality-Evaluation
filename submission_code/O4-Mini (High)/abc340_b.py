def main():
    import sys
    input = sys.stdin.readline
    
    Q = int(input())
    A = []
    outputs = []
    
    for _ in range(Q):
        parts = input().split()
        t = parts[0]
        if t == '1':
            x = int(parts[1])
            A.append(x)
        else:  # t == '2'
            k = int(parts[1])
            outputs.append(str(A[-k]))
    
    sys.stdout.write("
".join(outputs))

if __name__ == "__main__":
    main()