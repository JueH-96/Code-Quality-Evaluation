import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))
    
    A.sort()
    
    # Two pointers technique
    max_gifts = 0
    j = 0
    
    for i in range(N):
        while j < N and A[j] < A[i] + M:
            j += 1
        max_gifts = max(max_gifts, j - i)
    
    print(max_gifts)

if __name__ == "__main__":
    main()