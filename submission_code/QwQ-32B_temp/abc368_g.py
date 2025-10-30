import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    A = list(map(int, input[idx:idx+N]))
    idx +=N
    B = list(map(int, input[idx:idx+N]))
    idx +=N
    Q = int(input[idx]); idx +=1

    for _ in range(Q):
        query = input[idx]; idx +=1
        if query == '1':
            i = int(input[idx])-1; idx +=1
            x = int(input[idx]); idx +=1
            A[i] = x
        elif query == '2':
            i = int(input[idx])-1; idx +=1
            x = int(input[idx]); idx +=1
            B[i] = x
        elif query == '3':
            l = int(input[idx])-1; idx +=1
            r = int(input[idx])-1; idx +=1
            current = 0
            for i in range(l, r+1):
                current = max(current + A[i], current * B[i])
            print(current)

if __name__ == "__main__":
    main()