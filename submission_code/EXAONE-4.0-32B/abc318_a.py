import sys

def main():
    data = sys.stdin.readline().split()
    N = int(data[0])
    M = int(data[1])
    P = int(data[2])
    
    if M > N:
        print(0)
    else:
        k_max = (N - M) // P
        print(k_max + 1)

if __name__ == "__main__":
    main()