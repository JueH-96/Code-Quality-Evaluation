import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    T = n - k
    A.sort()
    ans = float('inf')
    for i in range(len(A) - T + 1):
        current_diff = A[i + T - 1] - A[i]
        if current_diff < ans:
            ans = current_diff
    print(ans)

if __name__ == "__main__":
    main()