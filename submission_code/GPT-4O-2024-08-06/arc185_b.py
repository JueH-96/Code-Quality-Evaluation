def can_make_non_decreasing(N, A):
    surplus = 0
    for i in range(N - 1):
        surplus += A[i] - max(A[i], A[i + 1] - surplus)
        if surplus < 0:
            return "No"
    return "Yes"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        A = list(map(int, data[index:index + N]))
        index += N
        results.append(can_make_non_decreasing(N, A))
    
    for result in results:
        print(result)