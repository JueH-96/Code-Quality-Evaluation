import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N, K = int(data[idx]), int(data[idx+1])
        idx +=2
        A = list(map(int, data[idx:idx+N]))
        idx +=N
        B = list(map(int, data[idx:idx+N]))
        idx +=N
        
        possible = True
        for b in B:
            if b not in A:
                possible = False
                break
        print("Yes" if possible else "No")

if __name__ == '__main__':
    main()