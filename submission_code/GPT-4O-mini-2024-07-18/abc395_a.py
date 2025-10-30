def is_strictly_increasing(N, A):
    for i in range(N - 1):
        if A[i] >= A[i + 1]:
            return "No"
    return "Yes"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    A = list(map(int, data[1].split()))
    
    result = is_strictly_increasing(N, A)
    print(result)